# dataloader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer

class DataLoaderCreator:
    def __init__(self, max_length=512, batch_size=8):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length
        self.batch_size = batch_size

    def load_data(self, file_path):
        """加载文本和标签"""
        data, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                data.append(text)
                labels.append(label)
        return data, labels

    def encode_labels(self, labels, categories):
        """将文本标签转换为 one-hot 编码"""
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)
        encoded_labels = label_encoder.transform(labels)
        return np.eye(len(categories))[encoded_labels]  # One-hot

    def tokenize_data(self, texts):
        """对文本进行 Tokenization 和 Padding"""
        return self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)

    def create_dataloader(self, encodings, labels):
        """创建 PyTorch DataLoader"""
        class NewsDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        dataset = NewsDataset(encodings, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def prepare_data(self, train_file, val_file, test_file, categories):
        """加载并准备所有数据"""
        train_texts, train_labels = self.load_data(train_file)
        val_texts, val_labels = self.load_data(val_file)
        test_texts, test_labels = self.load_data(test_file)

        y_train = self.encode_labels(train_labels, categories)
        y_val = self.encode_labels(val_labels, categories)
        y_test = self.encode_labels(test_labels, categories)

        train_encodings = self.tokenize_data(train_texts)
        val_encodings = self.tokenize_data(val_texts)
        test_encodings = self.tokenize_data(test_texts)

        train_loader = self.create_dataloader(train_encodings, y_train)
        val_loader = self.create_dataloader(val_encodings, y_val)
        test_loader = self.create_dataloader(test_encodings, y_test)

        return train_loader, val_loader, test_loader
