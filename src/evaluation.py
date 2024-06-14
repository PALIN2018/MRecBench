#!/usr/bin/env python3
import json
import logging
import os
import pandas as pd
import argparse
from tqdm import tqdm
from utils.metrics4rec import evaluate_all
from utils.literal_similarity import (
    TfidfSimilarity,
    SimHashSimilarity,
    BM25Similarity,
    SameCharsSimilarity,
    SequenceMatcherSimilarity
)
import openpyxl  # Ensure this is installed for Excel handling

# Setup for command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Recommendations")
    parser.add_argument('--dataset_name', type=str, default='sports', help='Name of the dataset')
    parser.add_argument('--max_seq_len', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--root_path', type=str, required=False, help='Root path for the output files')
    parser.add_argument('--item_pool_path', type=str, required=False, help='Path to item pool json')
    parser.add_argument('--similarity_methods', nargs='+', default=['None'], help='List of similarity methods to evaluate')
    return parser.parse_args()

class RecommendationEvaluator:
    def __init__(self, processed_data_path, item_pool_path, similarity_method_name='tf-idf', min_recommendations=1):
        logging.basicConfig(level=logging.INFO)
        self.processed_data_path = processed_data_path
        self.item_pool_path = item_pool_path
        self.min_recommendations = min_recommendations
        self.similarity_method = self.get_similarity_method(similarity_method_name)
        self.processed_data = self.load_json_data(processed_data_path)
        self.item_pool = self.load_json_data(item_pool_path)
        self.log_initial_info(similarity_method_name)

    @staticmethod
    def load_json_data(file_path):
        """Load and return data from a JSON file."""
        with open(file_path, 'r') as file:
            return json.load(file)

    def get_similarity_method(self, method_name):
        """Return the similarity method object based on the method name."""
        similarity_methods = {
            'tf-idf': TfidfSimilarity(),
            'simhash': SimHashSimilarity(),
            'bm25': BM25Similarity(),
            'samechars': SameCharsSimilarity(),
            'sequencematcher': SequenceMatcherSimilarity()
        }
        return similarity_methods.get(method_name, None)

    def log_initial_info(self, similarity_method_name):
        """Log initial information about the evaluation."""
        logging.info(f"Similarity Method: {similarity_method_name}")
        logging.info(f"Number of items in the test dataset: {len(self.processed_data)}")
        logging.info(f"Size of the item pool: {len(self.item_pool)}")

    def prepare_evaluation_data(self):
        """Prepare prediction and ground truth data for evaluation."""
        predictions, ground_truth = {}, {}
        self.setup_similarity_model()

        for user_id in tqdm(self.processed_data, desc="Evaluating"):
            info = self.processed_data[user_id]
            ground_truth[user_id] = set(info['target']['titles'])
            recommendations = info['api_response'].get('recommendations', [])
            if recommendations:
                mapped_titles = self.map_recommendations_to_titles(recommendations)
                predictions[user_id] = {title: len(mapped_titles) - idx for idx, title in enumerate(mapped_titles)}
            else:
                predictions[user_id] = {f"Item_{i}": self.min_recommendations - i for i in range(self.min_recommendations)}
        return predictions, ground_truth

    def setup_similarity_model(self):
        """Set up the similarity model using the item pool titles."""
        if self.similarity_method is None:
            return
        corpus = [item['title'] for item in self.item_pool.values() if 'title' in item]
        self.similarity_method.add_corpus(corpus)

    def map_recommendations_to_titles(self, recommendations):
        """Map recommendations to item titles in the item pool based on text similarity."""
        if self.similarity_method is None:
            return recommendations
        res = self.similarity_method.most_similar(recommendations, topn=1)
        return [self.similarity_method.corpus[corpus_id] for q_id, c in res.items() for corpus_id, s in c.items()]

    def evaluate(self):
        """Evaluate recommendations and return metrics for different topks."""
        predictions, ground_truth = self.prepare_evaluation_data()
        # Assuming evaluate_all now returns a dictionary with topk keys
        metrics = {f'@{k}': evaluate_all(predictions, ground_truth, topk=k) for k in [1, 3, 5, 10, 20, 30]}
        return metrics

# New function to gather all directories to be evaluated
def find_evaluation_paths(root_path, containing):
    """
    Traverse the directory to find all paths to 'processed_data.json' files
    within directories whose names contain specified substrings.
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == 'processed_data.json' and any(contained in dirpath for contained in containing):
                yield os.path.join(dirpath, filename)

def main():
    args = parse_args()
    

    # Use arguments to replace hardcoded values
    dataset_name = args.dataset_name
    max_seq_len = args.max_seq_len
    root_path = './output/{}_{}'.format(dataset_name, max_seq_len)  # Adjust as necessary to the correct path

    item_pool_path = args.item_pool_path
    similarity_methods = args.similarity_methods
    similarity_methods = [None if method == 'None' else method for method in args.similarity_methods]

    # Initialize variables
    results = []

    # Logging to debug path discovery
    logging.basicConfig(level=logging.INFO)

    for file_path_data in find_evaluation_paths(root_path, ['prompts_r-']):
        logging.info(f"Found processed_data.json at: {file_path_data}")

        # Parse template_id and model_name from the directory name
        dir_name_components = os.path.basename(os.path.dirname(file_path_data)).split('_')
        template_id = dir_name_components[1]
        model_name = '_'.join(dir_name_components[2:])

        for similarity_name in similarity_methods:
            file_path_item_pool = f'./datasets/{dataset_name}/item_pool_full.json'

            evaluator = RecommendationEvaluator(file_path_data, file_path_item_pool, similarity_name)
            evaluation_metrics = evaluator.evaluate()

            logging.info(evaluation_metrics)  # Log the evaluation metrics for debugging
            result = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "max_seq_len": max_seq_len,
                "template_id": template_id,
                "similarity_method": similarity_name if similarity_name else "None"
            }

            for k in [1, 3, 5, 10, 20, 30]:
                result[f"NDCG@{k}"] = evaluation_metrics[f'@{k}'][1]['ndcg']
            for k in [1, 3, 5, 10, 20, 30]:
                result[f"Hits@{k}"] = evaluation_metrics[f'@{k}'][1]['hit']

            results.append(result)

    results_df = pd.DataFrame(results)
    results_file = "output/{}_{}_{}_hit_ndcg_results.xlsx".format(dataset_name, max_seq_len,'_'.join(args.similarity_methods))
    results_df.to_excel(results_file, index=False)
    logging.info(f"Results written to {results_file}")

if __name__ == "__main__":
    main()
