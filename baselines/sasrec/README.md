# SASREC Experiment Steps

## 1. Install the Environment
```bash
pip install -r requirements.txt
```

## 2. Download Data to the `data` Folder
Download data.tar.gz from the following link, then unzip the file into the root directory of git.
Data: https://drive.google.com/file/d/1FR0enWgGN39WWl7owLz_AKlua8eiqbEM/view?usp=sharing
```bash
tar -zxvf data.tar.gz
```
## 3. Start the Training Scripts
### 3.1 train sasRec with modal base
Please use the --modal command to change the training mode. The modal option can be selected from (img, img_title, title, title_desc). The --news command allows the selection of files produced by different multimodal models. The --user_test command can choose to test either 400 selected users or all users. If you want to change the dataset, you can modify the command's "beaury", and the dataset selections available are (beaury, clothing, sports, toys).
```bash
cd sasrec/sasrec_multi_modal
torchrun --nproc_per_node 1 --master_port 1234 run.py \
--root_data_dir MRecBench/ \
--dataset data/beauty \
--behaviors beauty_users.tsv \
--news beauty_items_gpt4.tsv \
--negs beauty_negs.tsv \
--test_users beauty_test_user_name.txt \
--mode train \
--item_tower modal \
--load_ckpt_name None \
--label_screen output_dir \
--logging_num 4 \
--testing_num 1 \
--l2_weight 0.1 \
--drop_rate 0.1 \
--batch_size 64 \
--lr 1e-3 \
--embedding_dim 512 \
--dnn_layer 8 \
--num_words_abstract 128 \
--bert_model_load bert_base_uncased \
--epoch 150 \
--fine_tune_lr 0 \
--user_test \'400\'
--modal "img_title"
```
### 3.2 train sasRec with ID base
```bash
cd sasrec/sasrec_id_base
torchrun --nproc_per_node 1 --master_port 1234 run.py \
--root_data_dir MRecBench/ \
--dataset data/beauty \
--behaviors beauty_users.tsv \
--news beauty_items.tsv \
--negs beauty_negs.tsv \
--test_users beauty_test_user_name.txt \
--mode train \
--item_tower modal \
--load_ckpt_name None \
--label_screen output_dir \
--logging_num 4 \
--testing_num 1 \
--l2_weight 0.1 \
--drop_rate 0.1 \
--batch_size 64 \
--lr 1e-3 \
--embedding_dim 512 \
--dnn_layer 8 \
--news_attributes title \
--bert_model_load bert_base_uncased \
--epoch 150 \
--fine_tune_lr 0 \
--user_test \'400\'
```