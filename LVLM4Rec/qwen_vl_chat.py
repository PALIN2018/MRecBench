# import json
# import time
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import base64
# import httpx
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
# import concurrent.futures

# class LVLMRecommender:
#     def __init__(self, dataset_name, max_seq_len, template_id, model_name, incremental_mode=False):
#         """
#         Initialize the LVLMRecommender with necessary parameters.
#         """
#         self.model_name = model_name
#         self.template_id = template_id
#         self.tokenizer, self.model = self.initialize_model(model_name)
#         self.input_file = f'./prompts/sampled_prompts/{dataset_name}_{max_seq_len}/prompts_{template_id}.json'
#         self.output_path = f'./results/{dataset_name}_{max_seq_len}/prompts_{template_id}_{model_name}/'

#     def initialize_model(self, model_name):
#         """
#         Initialize the qwen-vl-chat model and tokenizer.
#         """
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             trust_remote_code=True,
#             torch_dtype=torch.float16  # Use float16 for efficiency
#         )
#         model.eval()
#         return tokenizer, model

#     def read_json(self, file_path):
#         """
#         Read and return data from a JSON file.
#         """
#         with open(file_path, 'r') as file:
#             return json.load(file)

#     def write_json(self, data, file_path):
#         """
#         Write data to a JSON file.
#         """
#         with open(file_path, 'w') as file:
#             json.dump(data, file, indent=4, ensure_ascii=False)

#     def generate_response(self, content, image=None):
#         """
#         Generate a response using the qwen-vl-chat model.
#         """
#         if image:
#             # If an image is provided, process it and include in the input
#             inputs = self.tokenizer([content], return_tensors='pt')
#             image_input = self.prepare_image(image)
#             inputs.update({'images': image_input})
#         else:
#             inputs = self.tokenizer([content], return_tensors='pt')

#         # Move tensors to the appropriate device
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         # Generate response
#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, max_new_tokens=1024)
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

#     def prepare_image(self, image_url):
#         """
#         Download and preprocess the image.
#         """
#         response = httpx.get(image_url)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#         return image

#     def save_intermediate_results(self, processed_samples, failed_samples, processing_errors):
#         """
#         Save intermediate results to JSON files.
#         """
#         processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
#         if os.path.exists(processed_samples_path):
#             existing_processed_samples = self.read_json(processed_samples_path)
#             existing_processed_samples.update(processed_samples)
#             processed_samples = existing_processed_samples

#         failed_samples_path = os.path.join(self.output_path, 'failed_samples.json')
#         if os.path.exists(failed_samples_path):
#             existing_failed_samples = self.read_json(failed_samples_path)
#             existing_failed_samples.extend([sample for sample in failed_samples if sample not in existing_failed_samples])
#             failed_samples = existing_failed_samples

#         processing_errors_path = os.path.join(self.output_path, 'processing_errors.json')
#         if os.path.exists(processing_errors_path):
#             existing_processing_errors = self.read_json(processing_errors_path)
#             existing_processing_errors.update(processing_errors)
#             processing_errors = existing_processing_errors

#         self.write_json(processed_samples, processed_samples_path)
#         self.write_json(failed_samples, failed_samples_path)
#         self.write_json(processing_errors, processing_errors_path)
#         print('Sample Updated.')

#     def process_sample(self, sample_data, timeout_duration=120):
#         """
#         # Process a single sample by generating a response using the model
#         """
#         content = sample_data['prompt']
#         image = None
#         if self.template_id in ["s-1-image", "s-1-title-image", "s-2", "s-3"]:
#             image_url = sample_data['history']['online_combined_image_path']
#             image = image_url

#         # Generate response
#         response_text = self.generate_response(content, image=image)

#         # Assuming the model outputs a JSON string, parse it
#         # If not, adjust this part according to your model's output format
#         cleaned_content = response_text.replace('```json', '').replace('```', '').strip()
#         try:
#             return json.loads(cleaned_content)
#         except json.JSONDecodeError:
#             # If JSON decoding fails, return the raw text
#             return cleaned_content

#     def process_samples(self, prompts_data, max_retries=2, timeout_duration=120):
#         """
#         Process multiple samples and handle retries for failed samples.
#         """
#         processed_samples = {}
#         failed_samples = []
#         processing_errors = {}

#         sample_counter = 0
#         for sample_id, sample_data in tqdm(prompts_data.items(), desc="Processing samples"):
#             try:
#                 api_response = self.process_sample(sample_data)
#                 sample_data['api_response'] = api_response
#                 processed_samples[sample_id] = sample_data

#                 sample_counter += 1
#                 if sample_counter % 1 == 0:
#                     self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

#             except Exception as e:
#                 failed_samples.append(sample_id)
#                 processing_errors[sample_id] = {'error': str(e)}

#                 for _ in range(max_retries):
#                     try:
#                         api_response = self.process_sample(sample_data)
#                         sample_data['api_response'] = api_response
#                         processed_samples[sample_id] = sample_data
#                         failed_samples.remove(sample_id)
#                         if sample_counter % 1 == 0:
#                             self.save_intermediate_results(processed_samples, failed_samples, processing_errors)
#                         break
#                     except Exception as retry_error:
#                         processing_errors[sample_id] = {'retry_error': str(retry_error)}
#         self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

#         return processed_samples, failed_samples, processing_errors

#     def process_samples_and_save(self, resume_from_last=False, debug_mode=False):
#         """
#         Process samples and save the results to JSON files.
#         """
#         prompts_data = self.read_json(self.input_file)

#         if debug_mode:
#             print("Debug mode is ON: Processing only the first 3 samples.")
#             prompts_data = dict(list(prompts_data.items())[:3])

#         # Load previously processed samples if resume_from_last is True
#         processed_samples = {}
#         if resume_from_last:
#             processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
#             if os.path.exists(processed_samples_path):
#                 processed_samples = self.read_json(processed_samples_path)
#                 print(f"Resuming from last session. {len(processed_samples)} samples already processed.")

#         # Initialize the set of already processed sample IDs
#         already_processed_ids = set(processed_samples.keys())

#         # Create the output directory if it does not exist
#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)

#         start_time = time.time()

#         # Process samples
#         new_processed_samples, failed_samples, processing_errors = self.process_samples(
#             {k: v for k, v in prompts_data.items() if k not in already_processed_ids}
#         )

#         # Update the processed_samples dictionary with new data
#         processed_samples.update(new_processed_samples)

#         # Save the results
#         self.write_json(processed_samples, os.path.join(self.output_path, 'processed_data.json'))
#         self.write_json(failed_samples, os.path.join(self.output_path, 'failed_samples.json'))
#         self.write_json(processing_errors, os.path.join(self.output_path, 'processing_errors.json'))

#         end_time = time.time()
#         total_time = end_time - start_time
#         average_time_per_sample = total_time / len(prompts_data)

#         return self.output_path, total_time, average_time_per_sample

# # 配置参数
# dataset_name = "toys"
# max_seq_len = 10
# template_id = "s-1-image"
# model_name = "Qwen/Qwen-VL-Chat"  # 使用qwen-vl-chat模型

# def run_task(dataset_name, template_id, model_name, debug_mode):
#     """
#     Execute a single processing task using LVLMRecommender based on the given parameters.
#     """
#     # Instantiate the LVLMRecommender class with the given configuration parameters
#     processor = LVLMRecommender(dataset_name, max_seq_len, template_id, model_name)
#     # Call the process_samples_and_save method to process data and save the results
#     result = processor.process_samples_and_save(resume_from_last=True, debug_mode=debug_mode)
#     # Print task-related information
#     print(f"Dataset: {dataset_name}, Template ID: {template_id}, Total time taken: {result[1]:.2f} seconds")
#     print(f"Dataset: {dataset_name}, Template ID: {template_id}, Average time per sample: {result[2]:.2f} seconds")
#     return result

# # 定义任务列表
# tasks = [
#     ('toys', 's-1-image', 'Qwen/Qwen-VL-Chat', False),
#     # 添加更多任务，如果需要
# ]

# # 执行任务
# with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
#     # Start all tasks using a list comprehension
#     futures = [executor.submit(run_task, ds_name, tpl_id, model_name, debug_mode) for ds_name, tpl_id, model_name, debug_mode in tasks]

#     # Print the result of each future as it completes
#     for future in concurrent.futures.as_completed(futures):
#         future.result()



# import json
# import time
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import httpx
# from tqdm import tqdm
# import concurrent.futures

# class LVLMRecommender:
#     def __init__(self, dataset_name, max_seq_len, template_id, model_name, incremental_mode=False):
#         """
#         Initialize the LVLMRecommender with necessary parameters.
#         """
#         self.model_name = model_name
#         self.template_id = template_id
#         self.tokenizer, self.model = self.initialize_model(model_name)
#         self.input_file = f'./prompts/sampled_prompts/{dataset_name}_{max_seq_len}/prompts_{template_id}.json'
#         self.output_path = f'./results/{dataset_name}_{max_seq_len}/prompts_{template_id}_{model_name}/'

#     def initialize_model(self, model_name):
#     # Set random seed for reproducibility
#         torch.manual_seed(1234)

#         # Load the tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#         # Determine device
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True,
#                 torch_dtype=torch.float16,
#                 device_map=None,  # Do not use 'auto' to prevent meta tensors
#                 low_cpu_mem_usage=False  # Ensure all weights are loaded
#             ).to(device).eval()
#         else:
#             device = torch.device("cpu")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True,
#                 device_map=None,
#                 low_cpu_mem_usage=False
#             ).to(device).eval()

#         return tokenizer, model



#     def read_json(self, file_path):
#         """
#         Read and return data from a JSON file.
#         """
#         with open(file_path, 'r', encoding='utf-8') as file:
#             return json.load(file)

#     def write_json(self, data, file_path):
#         """
#         Write data to a JSON file.
#         """
#         with open(file_path, 'w', encoding='utf-8') as file:
#             json.dump(data, file, indent=4, ensure_ascii=False)

#     def generate_response(self, sample_data):
#         """
#         Generate a response using the Qwen-VL-Chat model.
#         """
#         content = sample_data['prompt']
#         image_url = None
#         if self.template_id in ["s-1-image", "s-1-title-image", "s-2", "s-3"]:
#             image_url = sample_data['history']['online_combined_image_path']

#         if image_url:
#             # Directly pass the image URL to tokenizer.from_list_format
#             query = self.tokenizer.from_list_format([
#                 {'image': image_url},
#                 {'text': content},
#             ])
#         else:
#             query = content

#         # Generate response using the model's chat method
#         with torch.no_grad():
#             response, history = self.model.chat(self.tokenizer, query=query, history=None)

#         return response

#     def save_intermediate_results(self, processed_samples, failed_samples, processing_errors):
#         """
#         Save intermediate results to JSON files.
#         """
#         processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
#         if os.path.exists(processed_samples_path):
#             existing_processed_samples = self.read_json(processed_samples_path)
#             existing_processed_samples.update(processed_samples)
#             processed_samples = existing_processed_samples

#         failed_samples_path = os.path.join(self.output_path, 'failed_samples.json')
#         if os.path.exists(failed_samples_path):
#             existing_failed_samples = self.read_json(failed_samples_path)
#             existing_failed_samples.extend([sample for sample in failed_samples if sample not in existing_failed_samples])
#             failed_samples = existing_failed_samples

#         processing_errors_path = os.path.join(self.output_path, 'processing_errors.json')
#         if os.path.exists(processing_errors_path):
#             existing_processing_errors = self.read_json(processing_errors_path)
#             existing_processing_errors.update(processing_errors)
#             processing_errors = existing_processing_errors

#         self.write_json(processed_samples, processed_samples_path)
#         self.write_json(failed_samples, failed_samples_path)
#         self.write_json(processing_errors, processing_errors_path)
#         print('Sample Updated.')

#     def process_sample(self, sample_data, timeout_duration=120):
#         """
#         Process a single sample by generating a response using the model.
#         """
#         try:
#             response_text = self.generate_response(sample_data)

#             # Assuming the model outputs a JSON string, parse it
#             # If not, adjust this part according to your model's output format
#             cleaned_content = response_text.strip()
#             try:
#                 return json.loads(cleaned_content)
#             except json.JSONDecodeError:
#                 # If JSON decoding fails, return the raw text
#                 return cleaned_content
#         except Exception as e:
#             raise e

#     def process_samples(self, prompts_data, max_retries=2, timeout_duration=120):
#         """
#         Process multiple samples and handle retries for failed samples.
#         """
#         processed_samples = {}
#         failed_samples = []
#         processing_errors = {}

#         sample_counter = 0
#         for sample_id, sample_data in tqdm(prompts_data.items(), desc="Processing samples"):
#             try:
#                 api_response = self.process_sample(sample_data)
#                 sample_data['api_response'] = api_response
#                 processed_samples[sample_id] = sample_data

#                 sample_counter += 1
#                 if sample_counter % 1 == 0:
#                     self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

#             except Exception as e:
#                 failed_samples.append(sample_id)
#                 processing_errors[sample_id] = {'error': str(e)}

#                 for _ in range(max_retries):
#                     try:
#                         api_response = self.process_sample(sample_data)
#                         sample_data['api_response'] = api_response
#                         processed_samples[sample_id] = sample_data
#                         failed_samples.remove(sample_id)
#                         if sample_counter % 1 == 0:
#                             self.save_intermediate_results(processed_samples, failed_samples, processing_errors)
#                         break
#                     except Exception as retry_error:
#                         processing_errors[sample_id] = {'retry_error': str(retry_error)}
#         self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

#         return processed_samples, failed_samples, processing_errors

#     def process_samples_and_save(self, resume_from_last=False, debug_mode=False):
#         """
#         Process samples and save the results to JSON files.
#         """
#         prompts_data = self.read_json(self.input_file)

#         if debug_mode:
#             print("Debug mode is ON: Processing only the first 3 samples.")
#             prompts_data = dict(list(prompts_data.items())[:3])

#         # Load previously processed samples if resume_from_last is True
#         processed_samples = {}
#         if resume_from_last:
#             processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
#             if os.path.exists(processed_samples_path):
#                 processed_samples = self.read_json(processed_samples_path)
#                 print(f"Resuming from last session. {len(processed_samples)} samples already processed.")

#         # Initialize the set of already processed sample IDs
#         already_processed_ids = set(processed_samples.keys())

#         # Create the output directory if it does not exist
#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)

#         start_time = time.time()

#         # Process samples
#         new_processed_samples, failed_samples, processing_errors = self.process_samples(
#             {k: v for k, v in prompts_data.items() if k not in already_processed_ids}
#         )

#         # Update the processed_samples dictionary with new data
#         processed_samples.update(new_processed_samples)

#         # Save the results
#         self.write_json(processed_samples, os.path.join(self.output_path, 'processed_data.json'))
#         self.write_json(failed_samples, os.path.join(self.output_path, 'failed_samples.json'))
#         self.write_json(processing_errors, os.path.join(self.output_path, 'processing_errors.json'))

#         end_time = time.time()
#         total_time = end_time - start_time
#         average_time_per_sample = total_time / len(prompts_data)

#         return self.output_path, total_time, average_time_per_sample

# # 配置参数
# dataset_name = "toys"
# max_seq_len = 10
# template_id = "s-1-image"
# model_name = "Qwen/Qwen-VL-Chat"  # 使用Qwen-VL-Chat模型

# def run_task(dataset_name, template_id, model_name, debug_mode):
#     """
#     Execute a single processing task using LVLMRecommender based on the given parameters.
#     """
#     # Instantiate the LVLMRecommender class with the given configuration parameters
#     processor = LVLMRecommender(dataset_name, max_seq_len, template_id, model_name)
#     # Call the process_samples_and_save method to process data and save the results
#     result = processor.process_samples_and_save(resume_from_last=True, debug_mode=debug_mode)
#     # Print task-related information
#     print(f"Dataset: {dataset_name}, Template ID: {template_id}, Total time taken: {result[1]:.2f} seconds")
#     print(f"Dataset: {dataset_name}, Template ID: {template_id}, Average time per sample: {result[2]:.2f} seconds")
#     return result

# # 定义任务列表
# tasks = [
#     ('toys', 's-1-image', 'Qwen/Qwen-VL-Chat', False),
#     # 添加更多任务，如果需要
# ]

# # 执行任务
# if __name__ == "__main__":
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
#         # Start all tasks using a list comprehension
#         futures = [executor.submit(run_task, ds_name, tpl_id, model_name, debug_mode) for ds_name, tpl_id, model_name, debug_mode in tasks]

#         # Print the result of each future as it completes
#         for future in concurrent.futures.as_completed(futures):
#             future.result()


import json
import time
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import concurrent.futures

class LVLMRecommender:
    def __init__(self, dataset_name, max_seq_len, template_id, model_name, incremental_mode=False):
        """
        Initialize the LVLMRecommender with necessary parameters.
        """
        self.model_name = model_name
        self.template_id = template_id
        self.tokenizer, self.model = self.initialize_model(model_name)
        self.input_file = f'./prompts/sampled_prompts/{dataset_name}_{max_seq_len}/prompts_{template_id}.json'
        self.output_path = f'./results/{dataset_name}_{max_seq_len}/prompts_{template_id}_{model_name}/'

    def initialize_model(self, model_name):
        """
        Initialize the Qwen-VL-Chat model and tokenizer.
        """
        # Set random seed for reproducibility
        torch.manual_seed(1234)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine device and load the model accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=None,  # Do not use 'auto' to prevent meta tensors
                low_cpu_mem_usage=False  # Ensure all weights are loaded
            ).to(device).eval()
        else:
            device = torch.device("cpu")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=False
            ).to(device).eval()

        return tokenizer, model

    def read_json(self, file_path):
        """
        Read and return data from a JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def write_json(self, data, file_path):
        """
        Write data to a JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def generate_response(self, sample_data):
        """
        Generate a response using the Qwen-VL-Chat model.
        """
        content = sample_data['prompt']
        image_url = None
        if self.template_id in ["s-1-image", "s-1-title-image", "s-2", "s-3"]:
            image_url = sample_data['history']['online_combined_image_path']

        if image_url:
            # Directly pass the image URL to tokenizer.from_list_format
            query = self.tokenizer.from_list_format([
                {'image': image_url},
                {'text': content},
            ])
        else:
            query = content

        # Generate response using the model's chat method
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, query=query, history=None)

        return response

    def save_intermediate_results(self, processed_samples, failed_samples, processing_errors):
        """
        Save intermediate results to JSON files.
        """
        processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
        if os.path.exists(processed_samples_path):
            existing_processed_samples = self.read_json(processed_samples_path)
            existing_processed_samples.update(processed_samples)
            processed_samples = existing_processed_samples

        failed_samples_path = os.path.join(self.output_path, 'failed_samples.json')
        if os.path.exists(failed_samples_path):
            existing_failed_samples = self.read_json(failed_samples_path)
            existing_failed_samples.extend([sample for sample in failed_samples if sample not in existing_failed_samples])
            failed_samples = existing_failed_samples

        processing_errors_path = os.path.join(self.output_path, 'processing_errors.json')
        if os.path.exists(processing_errors_path):
            existing_processing_errors = self.read_json(processing_errors_path)
            existing_processing_errors.update(processing_errors)
            processing_errors = existing_processing_errors

        self.write_json(processed_samples, processed_samples_path)
        self.write_json(failed_samples, failed_samples_path)
        self.write_json(processing_errors, processing_errors_path)
        print('Sample Updated.')

    def process_sample(self, sample_data, timeout_duration=120):
        """
        Process a single sample by generating a response using the model.
        """
        try:
            response_text = self.generate_response(sample_data)

            # Assuming the model outputs a JSON string, parse it
            # If not, adjust this part according to your model's output format
            cleaned_content = response_text.strip()
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                # If JSON decoding fails, return the raw text
                return cleaned_content
        except Exception as e:
            raise e

    def process_samples(self, prompts_data, max_retries=2, timeout_duration=120):
        """
        Process multiple samples and handle retries for failed samples.
        """
        processed_samples = {}
        failed_samples = []
        processing_errors = {}

        sample_counter = 0
        for sample_id, sample_data in tqdm(prompts_data.items(), desc="Processing samples"):
            try:
                api_response = self.process_sample(sample_data)
                sample_data['api_response'] = api_response
                processed_samples[sample_id] = sample_data

                sample_counter += 1
                if sample_counter % 1 == 0:
                    self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

            except Exception as e:
                failed_samples.append(sample_id)
                processing_errors[sample_id] = {'error': str(e)}

                for _ in range(max_retries):
                    try:
                        api_response = self.process_sample(sample_data)
                        sample_data['api_response'] = api_response
                        processed_samples[sample_id] = sample_data
                        failed_samples.remove(sample_id)
                        if sample_counter % 1 == 0:
                            self.save_intermediate_results(processed_samples, failed_samples, processing_errors)
                        break
                    except Exception as retry_error:
                        processing_errors[sample_id] = {'retry_error': str(retry_error)}
        self.save_intermediate_results(processed_samples, failed_samples, processing_errors)

        return processed_samples, failed_samples, processing_errors

    def process_samples_and_save(self, resume_from_last=False, debug_mode=False):
        """
        Process samples and save the results to JSON files.
        """
        prompts_data = self.read_json(self.input_file)

        if debug_mode:
            print("Debug mode is ON: Processing only the first 3 samples.")
            prompts_data = dict(list(prompts_data.items())[:3])

        # Load previously processed samples if resume_from_last is True
        processed_samples = {}
        if resume_from_last:
            processed_samples_path = os.path.join(self.output_path, 'processed_data.json')
            if os.path.exists(processed_samples_path):
                processed_samples = self.read_json(processed_samples_path)
                print(f"Resuming from last session. {len(processed_samples)} samples already processed.")

        # Initialize the set of already processed sample IDs
        already_processed_ids = set(processed_samples.keys())

        # Create the output directory if it does not exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        start_time = time.time()

        # Process samples
        new_processed_samples, failed_samples, processing_errors = self.process_samples(
            {k: v for k, v in prompts_data.items() if k not in already_processed_ids}
        )

        # Update the processed_samples dictionary with new data
        processed_samples.update(new_processed_samples)

        # Save the results
        self.write_json(processed_samples, os.path.join(self.output_path, 'processed_data.json'))
        self.write_json(failed_samples, os.path.join(self.output_path, 'failed_samples.json'))
        self.write_json(processing_errors, os.path.join(self.output_path, 'processing_errors.json'))

        end_time = time.time()
        total_time = end_time - start_time
        average_time_per_sample = total_time / len(prompts_data) if len(prompts_data) > 0 else 0

        return self.output_path, total_time, average_time_per_sample

def run_task(args):
    """
    Execute a single processing task using LVLMRecommender based on the given parameters.
    """
    processor = LVLMRecommender(
        dataset_name=args.dataset_name,
        max_seq_len=args.max_seq_len,
        template_id=args.template_id,
        model_name=args.model_name
    )
    result = processor.process_samples_and_save(resume_from_last=True, debug_mode=args.debug_mode)
    print(f"Dataset: {args.dataset_name}, Template ID: {args.template_id}, Total time taken: {result[1]:.2f} seconds")
    print(f"Dataset: {args.dataset_name}, Template ID: {args.template_id}, Average time per sample: {result[2]:.2f} seconds")
    return result

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="LVLM-based Recommender Prompt Generation using Qwen-VL-Chat model.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset (e.g., 'toys', 'beauty', 'sports', 'clothing').")
    parser.add_argument('--max_seq_len', type=int, default=10, help="Maximum sequence length. Default is 10.")
    parser.add_argument('--template_id', type=str, required=True, help="Template ID to use (e.g., 's-1-image', 's-1-title-image', etc.).")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen-VL-Chat", help="Model name to use. Default is 'Qwen/Qwen-VL-Chat'.")
    parser.add_argument('--debug_mode', action='store_true', help="Enable debug mode to process only a few samples.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of concurrent workers. Default is 1.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Define tasks based on the number of workers
    # Here, each task corresponds to one combination of parameters
    # Since the parameters are passed via command line, we'll assume one task
    tasks = [args]  # For simplicity, each run corresponds to one task

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Start all tasks using a list comprehension
        futures = [executor.submit(run_task, task) for task in tasks]

        # Print the result of each future as it completes
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")
