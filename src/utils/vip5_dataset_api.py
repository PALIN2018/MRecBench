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



class VIP5Dataset(object):
    def __init__(self, root, dataset):
        super(VIP5Dataset, self).__init__()

        dataset_choices = ['toys', 'beauty', 'sports', 'clothing']
        if dataset not in dataset_choices:
            raise ValueError(f'Invalid dataset: [{dataset}]. Only {dataset_choices} are supported.')

        self.dataset = dataset

        self._dataset_path = os.path.join(root, dataset)

        self._mapping = _load_json(os.path.join(self._dataset_path, 'datamaps.json'))

        # purchase history sequence
        self._instances = _load_pickle(os.path.join(self._dataset_path, 'purchase_history.pkl'))
        self._exp = _load_pickle(os.path.join(self._dataset_path, 'exp_splits.pkl'))

        self._item2id = self._mapping['item2id']
        self._user2id = self._mapping['user2id']
        self._item_list = list(self._mapping['item2id'].keys())
        self._id2item = self._mapping['id2item']
        self._item2img = _load_pickle(os.path.join(self._dataset_path, 'item2img.pkl'))
        self._user2neg = _load_negative_txt(os.path.join(self._dataset_path, 'negative_samples.txt'))

        self.meta_data = []
        for meta in _parse(os.path.join(self._dataset_path, "meta.json.gz")):
            self.meta_data.append(meta)

        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i

    def user2id(self, user: str) -> str:
        return self._user2id[user]
    
    def item2id(self, item: str) -> str:
        return self._item2id[item]

    def id2item(self, item_id: str) -> str:
        return self._id2item[item_id]

    def item2img(self, item: str) -> str:
        return str(self._item2img[item]).split("/")[-1]

    def id2img(self, item_id: str) -> str:
        return self.item2img(self.id2item(item_id))

    def item_list(self) -> List[str]:
        return self._item_list

    def instances(self) -> List[List[str]]:
        return self._instances
    
    def explanations(self):
        organized_data = {}

        # Organize data by reviewerID and then by asin
        for split in ['train', 'val', 'test']:
            for entry in self._exp[split]:
                reviewer_id = self.user2id(entry['reviewerID'])
                asin = self.item2id(entry['asin'])

                if reviewer_id not in organized_data:
                    organized_data[reviewer_id] = {'train': {}, 'val': {}, 'test': {}}

                # Selecting specific fields for the asin
                asin_data = {field: entry[field] for field in ['overall', 'reviewTime', 'explanation', 'feature']}
                
                # Adding the asin data under the appropriate split
                organized_data[reviewer_id][split][asin] = asin_data

        return organized_data

    def id2meta(self, item_id: str):
        return self.meta_data[self.meta_dict[self.id2item(item_id)]]

    def item2meta(self, item: str):
        return self.id2meta(self.item2id(item))
