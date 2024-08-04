import pandas as pd
import gzip
import ast
import json
import gc
from collections import defaultdict
import random
from gensim.models import Word2Vec
import numpy as np
import string
import pickle
import gensim
import torch
##### prod2vec을 위한 세션데이터 생성
def preprocess_and_create_sessions(data, time_threshold=30):
    df = pd.DataFrame(data)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')

    # 세션 ID 생성
    df = df.sort_values(by=['user_id', 'timestamp_dt'])
    df['session_id'] = (df.groupby('user_id')['timestamp_dt']
                        .diff().gt(pd.Timedelta(minutes=time_threshold))
                        .cumsum())
    return df

def generate_sequences_with_metadata(df):
    sequences = []
    for session_id, session_df in df.groupby('session_id'):
        sequence = []
        for _, row in session_df.iterrows():
            # item_id = row['item_id']
            title = row['title']
            description = row['description']
            category = row['category']
            store = row['store']

            sequence.append(
                f"{title}_{description}_{category}_{store}"  # f"{item_id}_{title}_{description}_{category}_{store}"
            )
        sequences.append(sequence)
    return sequences

def train_word2vec_model(sequences, vector_size=100, window=5, min_count=1, workers=4):
    model = gensim.models.Word2Vec(sentences=sequences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   workers=workers,
                                   sg=1)  # Skip-gram 모델
    return model



def get_embedding(item_str, model):
    vector_size = model.vector_size
    if item_str in model.wv:
        return model.wv[item_str]  # numpy array
    else:
        return np.zeros(vector_size)  # Default embedding


def add_embeddings_to_df(df, word2vec_model):
    embeddings = []
    for _, row in df.iterrows():
        product_str = f"{row['title']}_{row['description']}_{row['category']}_{row['store']}"
        embedding_array = get_embedding(product_str, word2vec_model)
        embeddings.append(embedding_array.tolist())

    df['product_embedding'] = embeddings
    return df


def encode_column(column, pad=False):
    frequencies = column.value_counts(ascending=False)
    if pad:
        mapping = pd.Series(index=frequencies.index, data=range(1, len(frequencies) + 1))
    else:
        mapping = pd.Series(index=frequencies.index, data=range(len(frequencies)))
    encoded_column = column.map(mapping).fillna(0).astype(int)
    return encoded_column


def filter_n_core(df, n):
    while True:
        item_counts = df['item_id'].value_counts()
        user_counts = df['user_id'].value_counts()

        df = df[df['item_id'].isin(item_counts[item_counts >= n].index)]
        df = df[df['user_id'].isin(user_counts[user_counts >= n].index)]

        new_item_counts = df['item_id'].value_counts()
        new_user_counts = df['user_id'].value_counts()

        if (new_item_counts >= n).all() and (new_user_counts >= n).all():
            break
    return df


def load_meta_data(meta_file_path):
    meta_data = {}
    with gzip.open(meta_file_path, 'rt', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                entry = ast.literal_eval(line)
                json_entry = json.dumps(entry)
                entry = json.loads(json_entry)
                asin = entry.get('asin')
                if asin:
                    categories = entry.get('categories', [])
                    category = categories[0][1] if len(categories[0]) > 1 else (
                        categories[0][0] if len(categories[0]) == 1 else 'Baby')
                    title = entry.get('title', '')
                    description = entry.get('description', '')
                    meta_data[asin] = {
                        'store': entry.get('brand'),
                        'category': category,
                        'title': title,
                        'description': description
                    }
            except (ValueError, SyntaxError) as e:
                print(f"Error decoding line {line_number}: {e}")
                print(f"Offending line: {line}")
                break
    return meta_data


def merge_chunks(raw_review_file_path, meta_data, chunk_size=100000):
    chunk_list = []
    columns_to_keep = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    new_column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'store', 'category', 'title', 'description']

    with gzip.open(raw_review_file_path, 'rt', encoding='utf-8') as file:
        chunk = []
        for line in file:
            entry = json.loads(line)
            filtered_entry = {k: entry[k] for k in columns_to_keep}
            meta_info = meta_data.get(filtered_entry['asin'],
                                      {'store': None, 'category': 'Toys & Games', 'title': '', 'description': ''})
            filtered_entry.update(meta_info)
            chunk.append(filtered_entry)
            if len(chunk) >= chunk_size:
                chunk_df = pd.DataFrame(chunk)
                chunk_list.append(chunk_df)
                chunk = []
                gc.collect()

        if chunk:
            chunk_df = pd.DataFrame(chunk)
            chunk_list.append(chunk_df)
            del chunk, chunk_df
            gc.collect()

    final_df = pd.concat(chunk_list, ignore_index=True)
    final_df.columns = new_column_names
    return final_df


def add_average_ratings(df):
    average_ratings = df.groupby('item_id')['rating'].mean().reset_index()
    average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
    df = df.merge(average_ratings, on='item_id', how='left')
    return df


def calculate_unit_time(date, min_date):
    year_diff = date.year - min_date.year
    half_year = 0 if date.month <= 6 else 1
    return year_diff * 2 + half_year


def process_final_dataframe(df):
    df['user_encoded'] = encode_column(df['user_id'])
    df['item_encoded'] = encode_column(df['item_id'], pad=True)
    df['cat_encoded'] = encode_column(df['category'], pad=True)
    # df['store_encoded'] = encode_column(df['store'])

    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
    min_date = df['timestamp_dt'].min()
    df['unit_time'] = df['timestamp_dt'].apply(lambda x: calculate_unit_time(x, min_date))

    return df


def data_encoding(config):
    print("Loading meta data...")
    meta_data = load_meta_data(config.raw_meta_file_path)
    print("Merging raw review chunks...")
    final_df = merge_chunks(config.raw_review_file_path, meta_data)
    final_df = add_average_ratings(final_df)
    final_df = filter_n_core(final_df, 5)
    final_df = process_final_dataframe(final_df)
    print("Preprocessing and creating sessions...")
    session_data = preprocess_and_create_sessions(final_df)
    sequences = generate_sequences_with_metadata(session_data)
    session_data.to_csv('../dataset/Toys_and_Games/session_data.csv', encoding='utf-8-sig', index=False)
    print("Training Word2Vec model...")
    word2vec_model = train_word2vec_model(sequences)
    word2vec_model.save('product_word2vec_model.bin')
    word2vec_model = gensim.models.Word2Vec.load('product_word2vec_model.bin')
    print("Prod2vec embedding starts...")
    final_df = add_embeddings_to_df(final_df, word2vec_model)
    final_df.to_pickle(config.review_file_path)
    print("Processing complete.")
    print("min year", final_df['timestamp_dt'].dt.year.min())
    print("final df\n", final_df)
    print("columns", final_df.columns)
    print(
        f"# {config.dataset} >> len: {len(final_df):,} >> user: {final_df['user_encoded'].nunique():,} >> item: {final_df['item_encoded'].nunique():,}")

