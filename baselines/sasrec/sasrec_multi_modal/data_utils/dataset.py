import torch
from torch.utils.data import Dataset
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import random
import os


class BuildTrainDataset(Dataset):
    def __init__(self, u2seq, user_neg_dic, item_content, img_data, item_num, max_seq_len, modal):
        self.u2seq = u2seq # {user_id, history}
        self.item_content = item_content # item_num, 512
        self.img_data = img_data # item_num, 512
        self.item_num = item_num
        self.user_neg_dic = user_neg_dic
        self.max_seq_len = max_seq_len + 1
        self.modal = modal

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = self.user_neg_dic[user_id][:tokens_Len]
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items) # 2, max_seq_len
        sample_items = torch.LongTensor(np.array(sample_items)).transpose(0, 1) # max_seq_len, 2
        text_emb_items = self.item_content[sample_items] # max_seq_len, 2, 512
        if self.modal in ("title", "title_desc"):
            return text_emb_items, torch.FloatTensor(log_mask)
        elif self.modal == "img_title":
            img_emb_items = torch.tensor(self.img_data[sample_items]) # max_seq_len, 2, 512
            return torch.cat([text_emb_items, img_emb_items], dim=-1), torch.FloatTensor(log_mask)
        elif self.modal == "img":
            img_emb_items = torch.tensor(self.img_data[sample_items]) # max_seq_len, 2, 512
            return img_emb_items, torch.FloatTensor(log_mask)
        else:
            raise RuntimeError(f"error {self.modal} modal!!")

class BuildTestDataset(Dataset):
    def __init__(self, u2seq, user_neg_dic, item_content, img_data, item_num, max_seq_len, 
                 test_user_name_set, user_name_to_id, modal):
        self.u2seq = u2seq # {user_id, history}
        self.item_content = item_content # item_num, 512
        self.img_data = img_data
        self.item_num = item_num
        self.user_neg_dic = user_neg_dic
        self.max_seq_len = max_seq_len + 1
        self.modal = modal

        self.user_seqs = []
        test_user_id_set = set()
        for user_name in test_user_name_set:
            test_user_id_set.add(user_name_to_id[user_name])
        for user_id in u2seq:
            if os.environ["USE_TEST"] == "400":
                if user_id in test_user_id_set:
                    self.user_seqs.append((user_id, u2seq[user_id]))
            else:
                self.user_seqs.append((user_id, u2seq[user_id]))
        print("user_seqs length is", len(self.user_seqs))


    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        user_id, seq = self.user_seqs[index]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len # max_len-1, 
        input_seq = [0] * mask_len_head + seq[:-1] # max_len-1, 
        target = seq[-1]
        negs = self.user_neg_dic[user_id][:29]
        candidates = [target] + negs # 30, 

        input_seq = torch.LongTensor(np.array(input_seq))
        candidates = torch.LongTensor(np.array(candidates))
        candidates_id = candidates
        log_mask = torch.FloatTensor(log_mask)
        text_input_seq = self.item_content[input_seq]
        text_candidates = self.item_content[candidates]
        if self.modal in ("title", "title_desc"):
            input_seq = text_input_seq
            candidates = text_candidates
        elif self.modal == "img":
            img_input_seq = torch.tensor(self.img_data[input_seq])
            img_candidates = torch.tensor(self.img_data[candidates])
            input_seq = img_input_seq
            candidates = img_candidates
        elif self.modal == "img_title":
            img_input_seq = torch.tensor(self.img_data[input_seq])
            img_candidates = torch.tensor(self.img_data[candidates])
            input_seq = torch.cat([text_input_seq, img_input_seq], dim=-1)
            candidates = torch.cat([text_candidates, img_candidates], dim=-1)
        else:
            raise RuntimeError(f"error {self.modal} modal!!")
        return input_seq, candidates, log_mask, user_id, candidates_id


class BuildEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len,
                 item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
