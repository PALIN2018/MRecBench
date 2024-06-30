torchrun --nproc_per_node 1 --master_port 1234 run.py \
--root_data_dir ../../ \
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