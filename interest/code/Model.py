import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class LongTermInterestModule(nn.Module):
    def __init__(self, embedding_dim):
        super(LongTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.W_l = nn.Parameter(torch.Tensor(self.combined_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_l) 
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, 1)            
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, 2 * embedding_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)
    
    def forward(self, item_his_embeds, cat_his_embeds, user_embed):        
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)        
        h = torch.matmul(combined_embeds, self.W_l) 
        user_embed_transformed = self.user_transform(user_embed)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_l = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)
        return z_l

class MidTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MidTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.rnn = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.W_m = nn.Parameter(torch.Tensor(hidden_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_m)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, hidden_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, item_his_embeds, cat_his_embeds, user_embed):
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        o, _ = self.rnn(combined_embeds)
        h = torch.matmul(o, self.W_m)
        user_embed_transformed = self.user_transform(user_embed)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_m = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)
        return z_m

class ShortTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ShortTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim 
        self.rnn = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.W_s = nn.Parameter(torch.Tensor(hidden_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_s)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, hidden_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)

        # Initialize GRU weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, item_his_embeds, cat_his_embeds, user_embed):
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        o, _ = self.rnn(combined_embeds)
        h = torch.matmul(o, self.W_s)
        user_embed_transformed = self.user_transform(user_embed)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_s = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)
        return z_s
    
def long_term_interest_proxy(item_his_embeds, cat_his_embeds):
    """
    Calculate the long-term interest proxy using both item and category embeddings.
    """
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    p_l_t = torch.mean(combined_history_embeds, dim=1)
    return p_l_t

def mid_term_interest_proxy(item_his_embeds, cat_his_embeds, mid_lens):
    """
    Calculate the mid-term interest proxy using masking for variable lengths and both item and category embeddings.
    """
    device = item_his_embeds.device
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    max_len = combined_history_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(mid_lens), max_len) < mid_lens.unsqueeze(1).to(device)
    
    masked_history = combined_history_embeds * mask.unsqueeze(-1).type_as(combined_history_embeds)
    valid_counts = mask.sum(1, keepdim=True)
    
    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_m_t = masked_history.sum(1) / safe_valid_counts.type_as(combined_history_embeds)
    p_m_t = torch.nan_to_num(p_m_t, nan=0.0)

    return p_m_t

def short_term_interest_proxy(item_his_embeds, cat_his_embeds, short_lens):
    """
    Calculate the short-term interest proxy using both item and category embeddings.
    """
    device = item_his_embeds.device
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    max_len = combined_history_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(short_lens), max_len) < short_lens.unsqueeze(1).to(device)
    
    masked_history = combined_history_embeds * mask.unsqueeze(-1).type_as(combined_history_embeds)
    valid_counts = mask.sum(1, keepdim=True)
    
    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_s_t = masked_history.sum(1) / safe_valid_counts.type_as(combined_history_embeds)
    p_s_t = torch.nan_to_num(p_s_t, nan=0.0)

    return p_s_t


def bpr_loss(a, positive, negative):
    """
    Simplified BPR loss without the logarithm, for contrastive tasks.
    
    Parameters:
    - a: The embedding vector (z_l, z_m, or z_s).
    - positive: The positive proxy or representation (p_l or z_l for long-term, and similarly for mid and short-term).
    - negative: The negative proxy or representation (p_m or z_m for long-term, and similarly for mid and short-term).
    """
    pos_score = torch.sum(a * positive, dim=1)  
    neg_score = torch.sum(a * negative, dim=1)
    return F.softplus(neg_score - pos_score)

def calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s):
    """
    Calculate the overall contrastive loss L_con for a user at time t,
    which is the sum of L_lm (long-mid term contrastive loss) and L_ms (mid-short term contrastive loss).
    """
    # Loss for the long-term and mid-term interests pair
    L_lm = bpr_loss(z_l, p_l, p_m) + bpr_loss(p_l, z_l, z_m) + \
           bpr_loss(z_m, p_m, p_l) + bpr_loss(p_m, z_m, z_l)
    
    # Loss for the mid-term and short-term interests pair
    L_ms = bpr_loss(z_m, p_m, p_s) + bpr_loss(p_m, z_m, z_s) + \
           bpr_loss(z_s, p_s, p_m) + bpr_loss(p_s, z_s, z_m)
    
    # Overall contrastive loss
    L_con = L_lm + L_ms
    return L_con

class InterestFusionModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(InterestFusionModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.gru_l = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.gru_m = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        input_dim_for_alpha = 3 * hidden_dim 

        self.mlp_alpha_l = nn.Sequential(
            nn.Linear(input_dim_for_alpha, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp_alpha_m = nn.Sequential(
            nn.Linear(input_dim_for_alpha, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.mlp_pred = nn.Sequential(
            nn.Linear(embedding_dim * 4, output_dim),  
            nn.ReLU(),            
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, item_embeds, cat_embeds):
        # Combine item and category history embeddings
        combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        
        # Long-term history feature extraction
        h_l, _ = self.gru_l(combined_history_embeds)
        h_l = h_l[:, -1, :]  

        # Mid-term history feature extraction
        batch_size, seq_len, _ = combined_history_embeds.size()     
        masks = torch.arange(seq_len, device=mid_lens.device).expand(batch_size, seq_len) >= (seq_len - mid_lens.unsqueeze(1))   
        masked_embeddings = combined_history_embeds * masks.unsqueeze(-1).float()

        h_m, _ = self.gru_m(masked_embeddings)
        h_m = h_m[:, -1, :] 

        # Attention weights
        alpha_l = self.mlp_alpha_l(torch.cat((h_l, z_l, z_m), dim=1))
        alpha_m = self.mlp_alpha_m(torch.cat((h_m, z_m, z_s), dim=1))

        # Interest representation
        z_t = alpha_l * z_l + (1 - alpha_l) * alpha_m * z_m + (1 - alpha_l) * (1 - alpha_m) * z_s
        y_int = self.mlp_pred(torch.cat((z_t, item_embeds, cat_embeds), dim=1))

        return y_int

class BCELossModule(nn.Module):
    def __init__(self):
        super(BCELossModule, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, y_pred, y_true):
        """
        Calculate the Binary Cross-Entropy Loss.
        y_pred: Predicted labels for positive classes.
        y_true: True labels.
        """
        return self.loss_fn(y_pred, y_true)
    
class CAMP(nn.Module):
    def __init__(self, num_users, num_items, num_cats, Config):
        super(CAMP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, Config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, Config.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats + 1, Config.embedding_dim, padding_idx=0)
        self.long_term_module = LongTermInterestModule(Config.embedding_dim)
        self.mid_term_module = MidTermInterestModule(Config.embedding_dim, Config.hidden_dim)
        self.short_term_module = ShortTermInterestModule(Config.embedding_dim, Config.hidden_dim)
        self.interest_fusion_module = InterestFusionModule(Config.embedding_dim, Config.hidden_dim, Config.output_dim)
        self.bce_loss_module = BCELossModule()

    def forward(self, batch, item_to_cat_dict, device):
        user_ids = batch['user']
        item_ids = batch['item']
        cat_ids = batch['cat']
        items_history_padded = batch['item_his']        
        cats_history_padded = batch['cat_his']        
        mid_lens = batch['mid_len']
        short_lens = batch['short_len']
        neg_items_ids = batch['neg_items']
        neg_cats_ids = torch.tensor([[item_to_cat_dict[item.item()] for item in neg_items] for neg_items in neg_items_ids], device=device)

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        cat_embeds = self.cat_embedding(cat_ids)
        item_his_embeds = self.item_embedding(items_history_padded)
        cat_his_embeds = self.cat_embedding(cats_history_padded)
        neg_items_embeds = self.item_embedding(neg_items_ids)
        neg_cats_embeds = self.cat_embedding(neg_cats_ids)

        z_l = self.long_term_module(item_his_embeds, cat_his_embeds, user_embeds)
        z_m = self.mid_term_module(item_his_embeds, cat_his_embeds, user_embeds)
        z_s = self.short_term_module(item_his_embeds, cat_his_embeds, user_embeds)

        p_l = long_term_interest_proxy(item_his_embeds, cat_his_embeds)
        p_m = mid_term_interest_proxy(item_his_embeds, cat_his_embeds, mid_lens)
        p_s = short_term_interest_proxy(item_his_embeds, cat_his_embeds, short_lens)
        loss_con = calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s)

        y_int_pos = self.interest_fusion_module(item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, item_embeds, cat_embeds)        
        y_int_negs = torch.stack([
            self.interest_fusion_module(item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, neg_embed.squeeze(1), cat_embed.squeeze(1))
            for neg_embed, cat_embed in zip(neg_items_embeds.split(1, dim=1), neg_cats_embeds.split(1, dim=1))
        ], dim=1).squeeze(2)
        loss_bce_pos = self.bce_loss_module(y_int_pos, torch.ones_like(y_int_pos))
        loss_bce_neg = self.bce_loss_module(y_int_negs, torch.zeros_like(y_int_negs))
        loss_bce = loss_bce_pos + loss_bce_neg

        loss = loss_con + loss_bce
        return loss, y_int_pos, y_int_negs