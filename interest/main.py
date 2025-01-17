import random
import logging
import os
from datetime import datetime
import numpy as np
import argparse
import itertools

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from config import Config
from preprocess import load_df, create_dataloader
from Model import CAMP
from training_utils import train, evaluate, test, EarlyStopping

random.seed(2024) 
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=200,
                    help="training epochs")
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size for training")
parser.add_argument("--dropout_rate", type=float, default=0.5,
                    help="dropout rate for model")

parser.add_argument("--embedding_dim", type=int, default=64,
                    help="embedding size for embedding vectors")
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="size of the hidden layer embeddings")
parser.add_argument("--output_dim", type=int, default=1,
                    help="size of the output layer embeddings")

parser.add_argument("--gamma", type=float, default=0.95,
                    help="discount factor")
parser.add_argument("--k", type=int, default=20,
                    help="value of k for evaluation metrics")

parser.add_argument('--discrepancy_loss_weight', type=float, default=0.01, 
                    help='Loss weight for discrepancy between long and short term user embedding.')
parser.add_argument('--regularization_weight', type=float, default=0.0001, 
                    help='weight for L2 regularization applied to model parameters')

parser.add_argument("--dataset", type=str, default='14_Toys',
                    help="dataset file name")
parser.add_argument("--data_type", type=str, default='unif',
                    help="dataset split type(reg, unif, seq)")
parser.add_argument("--df_preprocessed", action="store_true",
                    help="flag to indicate if the dataframe has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument('--wo_con', action="store_true", 
                    help='flag to indicate if model has conformity module')
parser.add_argument('--wo_qlt', action="store_true", 
                    help='flag to indicate if model has quality module')

parser.add_argument('--cuda_device', type=str, help='CUDA device to use')


args = parser.parse_args()
config = Config(args=args)

def setup_logging(dataset_name, data_type, option):
    log_dir = os.path.abspath('./log')
    dataset_log_dir = os.path.join(log_dir, dataset_name)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    current_date = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(dataset_log_dir, f'{current_date}_{data_type}{option}.txt')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d')

def main():
    option = ''
    if config.wo_con and config.wo_qlt:
        option += '_wo_both'
    elif config.wo_con:
        option += '_wo_con'        
    elif config.wo_qlt:
        option += '_wo_qlt'
    else:
        option += '_full'

    if args.cuda_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    else:
        if option == '_full':
            os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        elif option == '_wo_con':
            os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        elif option == '_wo_qlt':
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
    setup_logging(config.dataset, config.data_type, option)
        
    print(f"Data preprocessing for dataset {config.dataset}......")
    train_df, valid_df, test_df, num_users, num_items, num_cats = load_df(config)

    print("Create datasets......")
    train_loader, valid_loader, test_loader = create_dataloader(train_df, valid_df, test_df)

    del train_df, valid_df, test_df
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    if config.dataset == '14_Sports':
        if config.data_type == "unif":      
            learning_rates = [0.0005] 
            batch_sizes = [128]
            embedding_dims = [128]        
        # elif config.data_type == "seq":
        #     learning_rates = [0.001] 
        #     batch_sizes = [128]
        #     embedding_dims = [64]
        # elif config.data_type == "reg":  
        #     learning_rates = [0.001]
        #     batch_sizes = [64]
        #     embedding_dims = [64]
        else:
            learning_rates = [0.001, 0.0005, 0.0001]
            batch_sizes = [64, 128]
            embedding_dims = [64, 128]
    elif config.dataset == '14_Toys':
        if config.data_type == "unif":      
            learning_rates = [0.0005] 
            batch_sizes = [64]
            embedding_dims = [128]   
        # elif config.data_type == "seq":
        #     learning_rates = [0.0005] 
        #     batch_sizes = [64]
        #     embedding_dims = [128]
        # elif config.data_type == "reg": 
        #     learning_rates = [0.0005]
        #     batch_sizes = [64]
        #     embedding_dims = [64]
        else:
            learning_rates = [0.001, 0.0005, 0.0001]
            batch_sizes = [64, 128]
            embedding_dims = [64, 128]
    else:
        learning_rates = [0.0005]#0.001, 0.0005, 0.0001
        batch_sizes = [64]
        embedding_dims = [100] #64, 128,

    best_loss = float('inf')
    best_model_params = {}
    best_model = None

    if not config.test_only:
        for lr, batch_size, embedding_dim in itertools.product(learning_rates, batch_sizes, embedding_dims):            
            print(f"{config.dataset}_{config.data_type}{option}_with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")
            logging.info(f"{config.dataset}_{config.data_type}{option}_with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")
            
            config.lr = lr
            config.batch_size = batch_size
            config.embedding_dim = embedding_dim                      

            model = CAMP(num_users, num_items, num_cats, config).to(device)
            optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            early_stopping = EarlyStopping(patience=10, verbose=True)

            for epoch in range(config.num_epochs):
                train_loss = train(model, train_loader, optimizer, device)                            
                valid_loss = evaluate(model, valid_loader, device)
                scheduler.step()

                logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_params = {'lr': config.lr, 'batch_size': config.batch_size, 'embedding_dim': config.embedding_dim, 'epoch': epoch}
                    best_model = model.state_dict()

                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            

            # if config.dataset == "14_Sports":
            #     if config.data_type == "unif":
            #         inv_ratio = 0.5
            #     elif config.data_type == "reg":
            #         inv_ratio = 0.5
            # elif config.dataset == "14_Toys":
            #     if config.data_type == "unif":
            #         inv_ratio = 0.4
            #     elif config.data_type == "reg":
            #         inv_ratio = 0.2
            if option in ['_wo_con', '_wo_both']:
                inv_ratio = 1
                average_loss, results = test(model, test_loader, device, inv_ratio, k_list=[20])
                for k, metrics in results.items():
                    logging.info(f"{inv_ratio:.1f} [Test only] Test Loss: {average_loss:.4f}, Pre@{k}: {metrics['Precision']:.4f}, Rec@{k}: {metrics['Recall']:.4f}, NDCG@{k}: {metrics['NDCG']:.4f}, HR@{k}: {metrics['Hit Rate']:.4f}, AUC: {metrics['AUC']:.4f}, MRR: {metrics['MRR']:.4f}")
            else:
                for inv_ratio in np.linspace(0, 1, 11):
                    average_loss, results = test(model, test_loader, device, inv_ratio, k_list=[20])
                    for k, metrics in results.items():
                        logging.info(f"{inv_ratio:.1f} [Test only] Test Loss: {average_loss:.4f}, Pre@{k}: {metrics['Precision']:.4f}, Rec@{k}: {metrics['Recall']:.4f}, NDCG@{k}: {metrics['NDCG']:.4f}, HR@{k}: {metrics['Hit Rate']:.4f}, AUC: {metrics['AUC']:.4f}, MRR: {metrics['MRR']:.4f}")
            
            # Clear memory and cache after each run
            del model, optimizer, scheduler, early_stopping
            torch.cuda.empty_cache()

        if best_model is not None:
            print(f"Best Model Parameters: {best_model_params}")
            logging.info(f"Best Model Parameters: {best_model_params}")
            date_str = datetime.now().strftime('%y%m%d')
            config.embedding_dim = best_model_params['embedding_dim']
            model = CAMP(num_users, num_items, num_cats, config).to(device)
            model.load_state_dict(best_model)    
            final_save_path = f'{config.model_save_path}{date_str}_best_model_{config.data_type}{option}.pt'
            if not os.path.exists(os.path.dirname(config.model_save_path)):
                os.makedirs(os.path.dirname(config.model_save_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_dim': best_model_params['embedding_dim'],  # Save the best embedding dimension
                'lr': best_model_params['lr'],
                'batch_size': best_model_params['batch_size']
            }, final_save_path)
    else:
        date_str = input("Enter the date string of the saved model (format: yymmdd): ")
        input_data_type = input("Enter the data type of the saved model (format: reg, unif, seq): ")
        input_option = input("Enter the option of the saved model (format: full, wo_both, wo_con, wo_qlt): ")
        # date_str = '240618'
        # input_data_type = 'unif'
        # input_option = 'full'
        model_path = f'../model/{config.dataset}/{date_str}_best_model_{input_data_type}_{input_option}.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            config.embedding_dim = checkpoint['embedding_dim']
            model = CAMP(num_users, num_items, num_cats, config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
        
        for inv_ratio in np.linspace(0, 1, 11):
            average_loss, results = test(model, test_loader, device, inv_ratio, k_list=[5, 10, 20])
            for k, metrics in results.items():
                logging.info(f"{inv_ratio} [Test only] Test Loss: {average_loss:.4f}, Pre@{k}: {metrics['Precision']:.4f}, Rec@{k}: {metrics['Recall']:.4f}, NDCG@{k}: {metrics['NDCG']:.4f}, HR@{k}: {metrics['Hit Rate']:.4f}, AUC: {metrics['AUC']:.4f}, MRR: {metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()