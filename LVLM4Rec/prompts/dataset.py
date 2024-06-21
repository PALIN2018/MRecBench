import os
import gzip
import json
import pickle

from typing import List

def _load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def _load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def _parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def _load_negative_txt(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.split()
            user_id = elements[0]
            negative_items = elements[1:]
            data_dict[user_id] = negative_items
    return data_dict

class MRecDataset(object):
    def __init__(self, root, dataset):
        super(MRecDataset, self).__init__()

        dataset_choices = ['toys', 'beauty', 'sports', 'clothing']
        if dataset not in dataset_choices:
            raise ValueError(f'Invalid dataset: [{dataset}]. Only {dataset_choices} are supported.')

        self.dataset = dataset

        self._dataset_path = os.path.join(root, dataset)

        self._mapping = _load_json(os.path.join(self._dataset_path, 'datamaps.json'))
        self.datamaps = _load_json(os.path.join(self._dataset_path, 'datamaps.json'))

        self._item2side = _load_json(os.path.join(self._dataset_path, 'item2side.json'))
        self.item_pool_full = _load_json(os.path.join(self._dataset_path, 'item2side.json'))

        self._instances = _load_pickle(os.path.join(self._dataset_path, 'purchase_history.pkl'))

        self._item2id = self._mapping['item2id']
        self._user2id = self._mapping['user2id']
        self._item_list = list(self._mapping['item2id'].keys())
        self._id2item = self._mapping['id2item']
        self._user2neg = _load_negative_txt(os.path.join(self._dataset_path, 'negative_samples.txt'))

    def user2id(self, user: str) -> str:
        return self._user2id[user]
    
    def item2id(self, item: str) -> str:
        return self._item2id[item]
    
    def item2side(self, item: str) -> str:
        return self._item2side[item]

    def id2item(self, item_id: str) -> str:
        return self._id2item[item_id]
    
    def id2img(self, item_id: str) -> str:
        return self._item2side[item_id]['image']['image_path'].split("/")[-1]

    def item_list(self) -> List[str]:
        return self._item_list

    def instances(self) -> List[List[str]]:
        return self._instances
