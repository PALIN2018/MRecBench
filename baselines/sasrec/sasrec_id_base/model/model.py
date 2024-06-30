import torch
from torch import nn
from .encoders import Bert_Encoder, User_Encoder, MLP_Layers
from torch.nn.init import xavier_normal_, constant_


class Model(torch.nn.Module):

    def __init__(self, args, item_num):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len + 1
        self.item_embedding = nn.Embedding(item_num+1, args.word_embedding_dim)

        self.fc = MLP_Layers(word_embedding_dim=args.word_embedding_dim,
                                 item_embedding_dim=args.embedding_dim,
                                 layers=[args.embedding_dim] * (args.dnn_layer + 1),
                                 drop_rate=args.drop_rate)

        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward_test(self, input_seq, candidates, log_mask, local_rank):
        # 32, max_seq_len, 512
        # 32, 30, 512
        # 32, max_seq_len, 512
        input_seq = self.item_embedding(input_seq)
        candidates = self.item_embedding(candidates)
        batch_size, seq_len, word_dim = input_seq.shape
        _, neg_num, _ = candidates.shape
        input_seq = input_seq.view(-1, word_dim)
        input_seq = self.fc(input_seq)
        input_seq = input_seq.view(-1, seq_len, self.args.embedding_dim)
        input_seq = self.user_encoder(input_seq, log_mask, local_rank)
        input_emb = input_seq[:, -1, :].unsqueeze(1) # batch, 1, dim
        candidates = self.fc(candidates) # batch, 30, dim
        scores = (input_emb*candidates).sum(-1) # batch, 30
        return scores

    def forward(self, sample_items, log_mask, local_rank):
        # sample_items: batch*max_seq_len*2, 512
        sample_items = self.item_embedding(sample_items)
        input_embs_all = self.fc(sample_items) # batch*max_seq_len*2, dim
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim) # batch, max_seq_len, 2, dim
        pos_items_embs = input_embs[:, :, 0] # batch, max_seq_len, dim
        neg_items_embs = input_embs[:, :, 1] # batch, max_seq_len, dim

        input_logs_embs = pos_items_embs[:, :-1, :] # batch, max_seq_len-1, dim
        target_pos_embs = pos_items_embs[:, 1:, :] # batch, max_seq_len-1, dim
        target_neg_embs = neg_items_embs[:, :-1, :] # batch, max_seq_len-1, dim

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank) # batch, max_seq_len-1, dim
        pos_score = (prec_vec * target_pos_embs).sum(-1) # batch, max_seq_len-1
        neg_score = (prec_vec * target_neg_embs).sum(-1) # batch, max_seq_len-1

        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        return loss
