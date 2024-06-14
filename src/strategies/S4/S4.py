# -*- coding: utf-8 -*-
import os
import json
from openai import OpenAI
from tqdm import tqdm
from anthropic import Anthropic

import argparse

parser = argparse.ArgumentParser(description='Process some parameters.')

parser.add_argument('--img_type', type=str, required=True, help='Image type, e.g., beauty, sports, toys, clothing')
parser.add_argument('--des_path', type=str, required=True, help='Image description file path')
parser.add_argument('--item_pool_full_path', type=str, required=True, help='Path to item_pool_full.json')
parser.add_argument('--ori_prompt_path', type=str, required=True, help='S1 prompt file path. Should be r-2')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the result. Should be a .json file')
parser.add_argument('--save_path_failed', type=str, required=True, help='Path to save the failed result. Should be a .json file')
parser.add_argument('--datamap_path', type=str, required=True, help='Path to datamap.json')
parser.add_argument('--tsv_path', type=str, required=True, help='Path for the result of S2')
parser.add_argument('--base_img_path', type=str, required=True, help='Image file path')
parser.add_argument('--model_name', type=str, choices=['gpt4v', 'gpt4o', 'claude3'], required=True, help='Model name, e.g., gpt4v, gpt4o, claude3')
parser.add_argument('--model', type=str, required=True, help='Model identifier, like: "gpt-4-vision-preview"')

args = parser.parse_args()

img_type = args.img_type
des_path = args.des_path
ori_prompt_path = args.ori_prompt_path
save_path = args.save_path
save_path_failed = args.save_path_failed
model_name = args.model_name
model = args.model


def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def read_origin_result(path):
    if not os.path.exists(path):
        with open(path, 'w') as file:
            json.dump({}, file)
    return read_json(path)


# save
def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


img_id2des = read_json(des_path)  # key: id, value: dict
ori_prompts = read_json(ori_prompt_path)

def process_gpt(messages):
    api_key = ""
    api_base = ""
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0
    )

    if response.choices:
        message_content = response.choices[0].message.content
        cleaned_content = message_content.replace('```json', '').replace('```', '').strip()
        return cleaned_content
    else:
        raise Exception("No response from API")

    return ""

def process_claude3(messages):
    client = Anthropic(
        base_url="",
        auth_token=""
    )
    return client.messages.create(
        model=model,
        messages=messages,
        timeout=50,
        max_tokens=3000,
        temperature=0
    )

def process(messages):
    if model_name == 'gpt4v':
        return process_gpt(messages)
    elif model_name == 'gpt4o':
        return process_gpt(messages)
    elif model_name == 'claude3':
        return process_claude3(messages)

def run(debug=True):
    max_retry = 2
    cur_result = read_origin_result(save_path)
    cur_result_failed = read_origin_result(save_path_failed)

    count = 0
    for id in tqdm(ori_prompts.keys(), desc="Processing samples"):
        if id in cur_result and 'api_response' in cur_result[id] and cur_result[id]['api_response']:
            print(id + " already saved")
            continue
        purchased_items_id = ori_prompts[id]['history']['id']
        purchased_items_title = ori_prompts[id]['history']['titles']
        purchased_items_count = len(purchased_items_id)
        prompt_start = "The user has purchased {} items (including title and description) in chronological order:".format(
            purchased_items_count)
        title_des = ""
        for i, img_id in enumerate(purchased_items_id):
            title = purchased_items_title[i]
            des = img_id2des[img_id]["generated_description"]
            title_des += "(title: " + title + "| description: " + des + ")" + ","
        title_des = title_des[:-1] + "\n"
        prompt_start += title_des

        prompt_candidate = ori_prompts[id]['prompt'].split("\n")[1]
        prompt_end = "\nPlease rank these 30 candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history. \nDo not search on the Internet.\nOutput format: Please output directly in JSON format, including keys \"explanation\", and \"recommendations\". It follows this structure:\n- explanation: [Brief explanation of the rationale behind the recommendations]. \n- recommendations: [A list of 30 items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.]."

        prompt = prompt_start + prompt_candidate + prompt_end

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]

        recommand_res = ""
        id2failed = ""
        try:
            recommand_res = process(messages)
        except Exception as e:
            print(id + " error:", e)
            id2failed = str(e)
            for _ in range(max_retry):
                try:
                    recommand_res = process(messages)
                except Exception as retry_error:
                    id2failed = str(retry_error)
                    print(id + " retry error:", retry_error)
        data = {}
        data["prompt"] = prompt
        data["history"] = ori_prompts[id]['history']
        data["target"] = ori_prompts[id]["target"]
        if not recommand_res:
            print(id + " failed")
            cur_result_failed[id] = id2failed
            write_json(cur_result_failed, save_path_failed)
        else:
            print(id + " updated successfully")
            try:
                data['api_response'] = json.loads(recommand_res)
            except Exception as e:
                print(id + " json load failed!")
                data['api_response'] = recommand_res
            if id in cur_result_failed:
                cur_result_failed.pop(id)
                write_json(cur_result_failed, save_path_failed)

        # save
        cur_result[id] = data
        write_json(cur_result, save_path)

        count += 1
        if count >= 10 and debug:
            break

    print("done!")


run(False)