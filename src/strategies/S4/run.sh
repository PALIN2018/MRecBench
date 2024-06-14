#!/bin/bash

python S4.py \
  --img_type "beauty" \
  --des_path "/path/to/description" \
  --ori_prompt_path "/path/to/ori_prompt" \
  --save_path "/path/to/save.json" \
  --save_path_failed "/path/to/save_failed.json" \
  --model_name "gpt4v" \
  --model "gpt-4-vision-preview"
