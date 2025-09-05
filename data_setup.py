import json
import torch
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

def make_dummy_data(data_path, dataset, dummy_path, no_samples, dev_size, shuffle):
    dummy_path.mkdir(parents=True, exist_ok=True)
    data_dict = {'image': [], 'caption': [], 'label': []}
    annotations_path = data_path / 'annotations' / f'{dataset}.json'
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
        for annotation in annotations:
            curr_annotation = annotations[annotation]
            data_dict['image'].append(curr_annotation['image'])
            data_dict['caption'].append(curr_annotation['caption'])
            data_dict['label'].append(curr_annotation['label'])

    df = pd.DataFrame(data_dict)
    labels_count = df['label'].value_counts()
    unique_labels = labels_count.index
    no_labels = labels_count.sum()

    dummy_labels_count = {label: round(value/no_labels*no_samples) for label, value in zip(unique_labels, labels_count)}
    train_labels_count, dev_labels_count = {}, {}
    for label in dummy_labels_count:
        dev_labels_count[label] = round(dummy_labels_count[label]*dev_size)
        train_labels_count[label] = dummy_labels_count[label]-dev_labels_count[label]

    train_df, dev_df = pd.DataFrame(), pd.DataFrame()
    for label in unique_labels:
        train_df = pd.concat([train_df, df[df['label']==label].iloc[:train_labels_count[label]]], axis=0)
        dev_df = pd.concat([dev_df, df[df['label']==label].iloc[train_labels_count[label]:train_labels_count[label]+dev_labels_count[label]]], axis=0)
    train_df = train_df.sample(frac=1, ignore_index=True) if shuffle else train_df.reset_index()
    dev_df = dev_df.sample(frac=1, ignore_index=True) if shuffle else dev_df.reset_index()

    images_path = data_path / 'images' / dataset
    train_images_path = dummy_path / 'images' / 'train'
    dev_images_path = dummy_path / 'images' / 'dev'
    train_images_path.mkdir(parents=True, exist_ok=True)
    dev_images_path.mkdir(parents=True, exist_ok=True)

    train_dict, dev_dict = {}, {}
    for i in range(len(train_df)):
        curr_sample = train_df.iloc[i]
        train_dict[f'{i}'] = {'image': curr_sample['image'], 'caption': curr_sample['caption'], 'label': curr_sample['label']}
        shutil.copy(images_path/curr_sample['image'], train_images_path/curr_sample['image'])
    for i in range(len(dev_df)):
        curr_sample = dev_df.iloc[i]
        dev_dict[f'{i}'] = {'image': curr_sample['image'], 'caption': curr_sample['caption'], 'label': curr_sample['label']}
        shutil.copy(images_path/curr_sample['image'], dev_images_path/curr_sample['image'])
    annotations_path = dummy_path / 'annotations'
    annotations_path.mkdir(parents=True, exist_ok=True)
    with open(annotations_path / 'train.json', 'w') as f: f.write(json.dumps(train_dict, indent=4))
    with open(annotations_path / 'dev.json', 'w') as f: f.write(json.dumps(dev_dict, indent=4))

# function to read data file (.csv)
def preprocess_data(args, path, dataset, tokenizer, segmenter):
    data_dict = {'images': [], 'input_ids': [],
                 'attention_mask': [], 'labels': []}
    annotations_path = path / 'annotations' / f'{dataset}.json'
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
        for index, annotation in enumerate(annotations):
            caption = annotations[annotation]['caption']
            if args.WORD_SEG:
                tokens = segmenter(caption)
                caption = ""
                for token in tokens:
                    if "##" in token["word"]: caption += token["word"].replace("##","")
                    elif token["entity"] == "I": caption += "_" + token["word"]
                    else: caption += " " + token["word"]
                caption = caption.strip()
            tokenizer_ouput = tokenizer(caption, padding='max_length',
                                        truncation=True, max_length=args.PLM_MAX_TOKEN)
            data_dict['images'].append(annotations[annotation]['image'])
            data_dict['input_ids'].append(tokenizer_ouput.input_ids)
            data_dict['attention_mask'].append(tokenizer_ouput.attention_mask)
            data_dict['labels'].append(annotations[annotation]['label'] if annotations[annotation]['label'] is not None else 0)
            if (index+1)%500 == 0:
                print(f'Finished {index+1} samples')
    return pd.DataFrame(data_dict)

# function to get the labels in the data (different for task 1 and task 2)
def get_labels(df):
    labels = df['labels'].unique()
    labels_to_ids = {label:index for index, label in enumerate(labels)}
    ids_to_labels = {index:label for label, index in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels

# hope data class (pytorch custom dataset)
class SarcasmData(Dataset):
    def __init__(self, pvm_config, df, data_path, dataset):
        self.df = df
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(pvm_config['input_size'][1:]),
                                             transforms.Normalize(mean=pvm_config['mean'], std=pvm_config['std'])])
        self.images_path = data_path / 'images' / dataset
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image = plt.imread(self.images_path / self.df['images'].iloc[idx])
        if len(image.shape) != 3:
            image = np.tile(image[..., None], (1,1,3))
        data = {'pixel_values': self.transform(image),
                'input_ids': torch.tensor(self.df['input_ids'].iloc[idx]),
                'attention_mask': torch.tensor(self.df['attention_mask'].iloc[idx]),
                'labels': self.df['labels'].iloc[idx]}
        return data