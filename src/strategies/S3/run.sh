#!/bin/bash

python S3.py \
  --img_type "beauty" \
  --des_path "/path/to/description" \
  --item_pool_full_path "/path/to/item_pool_full.json" \
  --ori_prompt_path "/path/to/ori_prompt" \
  --save_path "/path/to/save.json" \
  --save_path_failed "/path/to/save_failed.json" \
  --datamap_path "/path/to/datamap.json" \
  --tsv_path "/path/to/result.tsv" \
  --base_img_path "/path/to/base_img" \
  --model_name "gpt4v" \
  --model "gpt-4-vision-preview"