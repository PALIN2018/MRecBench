import torch.optim as optim
import re
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig

from parameters import parse_args
from model import Model, Bert_Encoder
from data_utils import read_news, read_news_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings, get_item_word_embs, BuildTestDataset
from data_utils.utils import *
import random
import pandas as pd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args, local_rank):
    print('load bert model...')
    bert_model_load = args.bert_model_load
    tokenizer = BertTokenizer.from_pretrained(bert_model_load)
    config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(bert_model_load, config=config)
    os.environ["USE_TEST"] = args.user_test

    if 'tiny' in args.bert_model_load:
        args.word_embedding_dim = 128
    if 'mini' in args.bert_model_load:
        args.word_embedding_dim = 256
    if 'small' in args.bert_model_load:
        args.word_embedding_dim = 512
    if 'medium' in args.bert_model_load:
        args.word_embedding_dim = 512
    if 'base' in args.bert_model_load:
        args.word_embedding_dim = 768
    if 'large' in args.bert_model_load:
        args.word_embedding_dim = 1024

    for index, (name, param) in enumerate(bert_model.named_parameters()):
        param.requires_grad = False
    print('read news...')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)

    print('read behaviors...')
    item_num, item_id_to_dic, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, \
    user_name_to_id, user_neg_dic, item_id_before_to_now = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), 
                       os.path.join(args.root_data_dir, args.dataset, args.negs), before_item_id_to_dic,
                       before_item_name_to_id, before_item_id_to_name,
                       args.max_seq_len, args.min_seq_len, Log_file)
    print("read test users...")
    test_user_name_set = set()
    with open(os.path.join(args.root_data_dir, args.dataset, args.test_users)) as f:
        for line in f:
            test_user_name_set.add(line.strip())
    print(f"test user num is {len(test_user_name_set)}")

    print('combine news information...')
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask = get_doc_input_bert(item_id_to_dic, args)

    item_content = np.concatenate([
        x for x in
        [news_title, news_title_attmask,
         news_abstract, news_abstract_attmask,
         news_body, news_body_attmask]
        if x is not None], axis=1)

    # print('Bert Encoder...')
    # bert_encoder = Bert_Encoder(args=args, bert_model=bert_model).to(local_rank)

    # print('get bert output...')
    # item_word_embs = get_item_word_embs(bert_encoder, item_content, 512, args, local_rank) # item_num, 512

    print('build dataset...')
    train_dataset = BuildTrainDataset(u2seq=users_train, user_neg_dic=user_neg_dic, item_num=item_num,
                                      max_seq_len=args.max_seq_len)

    print('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    print('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=True, sampler=train_sampler)

    print('build model...')
    model = Model(args, item_num).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    checkpoint = None  # new
    ckpt_path = None  # new
    start_epoch = 0
    is_early_stop = True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("##### total_num {} #####".format(total_num))
    print("##### trainable_num {} #####".format(trainable_num))

    print('\n')
    print('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    scaler = torch.cuda.amp.GradScaler()
    Log_screen.info('{} train start'.format(args.label_screen))
    df_dic = {
        "epoch": [],
        "hit@1": [],
        "hit@3": [],
        "hit@5": [],
        "hit@10": [],
        "hit@20": [],
        "hit@30": [],
        "ndcg@1": [],
        "ndcg@3": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "ndcg@20": [],
        "ndcg@30": [],
    }
    df_dic = OrderedDict(df_dic)
    max_dic = None
    max_df = None
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        print('\n')
        print('epoch {} start'.format(now_epoch))
        print('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()
        train_dl.sampler.set_epoch(now_epoch)
        for data in train_dl:
            sample_items, log_mask = data
            sample_items, log_mask = sample_items.to(local_rank), log_mask.to(local_rank)
            sample_items = sample_items.view(-1, sample_items.size(-1))

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bz_loss = model(sample_items, log_mask, local_rank)
                loss += bz_loss.data.float()
            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                print('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss.data))
            batch_index += 1

        if not need_break and now_epoch%1==0:
            # print('')
            # max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
            #     run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
            #              model, item_word_embs, users_history_for_valid, users_valid, 512, item_num,
            #              args.mode, is_early_stop, local_rank)
            dic, result_df = run_test(users_test, users_history_for_test, item_num, 
                     local_rank, user_neg_dic, test_user_name_set, user_name_to_id, model, 
                     before_item_id_to_name, item_id_before_to_now)
            if not max_dic or max_dic["hit@1"] < dic["hit@1"]:
                max_dic = dic
                max_df = result_df
            df_dic["epoch"].append(now_epoch)
            for k in dic:
                df_dic[k].append(dic[k])
            model.train()
            # if need_save and dist.get_rank() == 0:
            #     save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)
        print('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        if need_break:
            break
    df = pd.DataFrame()
    for k in df_dic:
        df[k] = df_dic[k]
    result_path = os.path.join(args.root_data_dir, args.dataset.split("/")[-1], args.label_screen+".tsv")
    df.to_csv(result_path, sep="\t", index=None)
    result_df_path = os.path.join(args.root_data_dir, args.dataset.split("/")[-1], "test_result"+args.label_screen+".tsv")
    max_df.to_csv(result_df_path, sep="\t", index=None)
    print(result_path)
    ls = []
    for k in ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@30", 
              "hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "hit@30"]:
        ls.append(str(max_dic[k]))
    print(max_dic)
    print("\t".join(ls))
    # if dist.get_rank() == 0:
        # save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)
    # print('\n')
    # print('%' * 90)
    # print(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    # print(' early stop in epoch {}'.format(early_stop_epoch))
    # print('the End')
    # Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))

def hit_compute(scores):
    # 得到排序后的索引
    indices = np.argsort(-scores, axis=1)
    nums, _ = indices.shape
    dic = OrderedDict()
    dic["hit@1"] = np.sum(indices[:, :1] == 0)/nums
    dic["hit@3"] = np.sum(indices[:, :3] == 0)/nums
    dic["hit@5"] = np.sum(indices[:, :5] == 0)/nums
    dic["hit@10"] = np.sum(indices[:, :10] == 0)/nums
    dic["hit@20"] = np.sum(indices[:, :20] == 0)/nums
    dic["hit@30"] = np.sum(indices[:, :30] == 0)/nums

    return dic

def calculate_ndcg(scores, k):
    # 获取排序后的索引
    tmp_indices = np.argsort(-scores, axis=1)
    indices = [[None for _ in range(tmp_indices.shape[1])] for _ in range(tmp_indices.shape[0])]
    for i in range(tmp_indices.shape[0]):
        for j in range(tmp_indices.shape[1]):
            n = tmp_indices[i, j]
            indices[i][n] = j
    indices = np.array(indices)
    dcg = (indices[:, 0] < k) / np.log2(indices[:, 0]+2)
    
    # 计算 IDCG
    idcg = 1 / np.log2(2)
    
    # 计算 NDCG
    ndcg = dcg / idcg
    
    return np.mean(ndcg)

def run_test(users_test, users_history_for_test, item_num, local_rank, user_neg_dic, 
             test_user_name_set, user_name_to_id, model, before_item_id_to_name, item_id_before_to_now):
    print("start testing...")
    user_id_to_name = {}
    for k in user_name_to_id:
        user_id_to_name[user_name_to_id[k]] = k
    dataset = BuildTestDataset(users_test, user_neg_dic, item_num, args.max_seq_len, 
                               test_user_name_set, user_name_to_id)
    dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=4, shuffle=False)
    model.eval()
    scores = []
    users = []
    all_candidates = []
    for input_seq, candidates, log_mask, user_id in tqdm(dataloader):
        input_seq, candidates, log_mask = input_seq.to(local_rank), candidates.to(local_rank), log_mask.to(local_rank)
        # print(input_seq.shape, candidates.shape, log_mask.shape)
        tmp_scores = model.module.forward_test(input_seq, candidates, log_mask, local_rank).detach().cpu().numpy().tolist()
        scores.extend(tmp_scores)
        users.extend(user_id)
        all_candidates.append(candidates.detach().cpu().numpy())
    scores = np.array(scores) # num, 30
    index = np.argsort(scores, axis=1)
    num = len(scores)
    df = pd.DataFrame()
    item_ls_ls = []
    users = [user_id_to_name[int(u)] for u in users]
    all_candidates = np.concatenate(all_candidates, axis=0)
    for i in range(num):
        index_one = index[i]
        item_ls = []
        for a in index_one[::-1]:
            a = all_candidates[i, a]
            item_ls.append(before_item_id_to_name[item_id_before_to_now[a]])
        item_ls_ls.append(" ".join(item_ls))
    df["user"] = users
    df["items"] = item_ls_ls
    dic = hit_compute(scores)
    dic["ndcg@1"] = calculate_ndcg(scores, 1)
    dic["ndcg@3"] = calculate_ndcg(scores, 3)
    dic["ndcg@5"] = calculate_ndcg(scores, 5)
    dic["ndcg@10"] = calculate_ndcg(scores, 10)
    dic["ndcg@20"] = calculate_ndcg(scores, 20)
    dic["ndcg@30"] = calculate_ndcg(scores, 30)
    for k in dic:
        print(f"** {k}: {dic[k]:.2f}")
    print("end testing...")
    return dic, df


def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_word_embs, user_history, users_eval, batch_size, item_num,
             mode, is_early_stop, local_rank):
    eval_start_time = time.time()
    print('Validating...')
    item_embeddings = get_item_embeddings(model, item_word_embs, batch_size, args, local_rank)
    valid_Hit10 = eval_model(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank)
    report_time_eval(eval_start_time, Log_file)
    print('')
    need_break = False
    need_save = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True
    else:
        early_stop_count += 1
        if early_stop_count > 10:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    print("local_rank", local_rank)
    if local_rank == -1:
        local_rank = 0
    print("local_rank", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(12345)

    model_load = args.bert_model_load
    dir_label = str(args.item_tower) + f'_{model_load}'
    if args.dnn_layer == 0:
        log_paras = f'{args.item_tower}_{model_load}_bs_{args.batch_size}'\
                    f'_ed_{args.embedding_dim}_lr_{args.lr}'\
                    f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}'
    else:
        log_paras = f'{args.item_tower}_{model_load}_mlp_{args.dnn_layer}_bs_{args.batch_size}'\
                    f'_ed_{args.embedding_dim}_lr_{args.lr}'\
                    f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr}'
    model_dir = os.path.join('./checkpoint_' + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)
    print(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(args)
    if 'train' in args.mode:
        train(args, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    print("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
