import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file):
        # read the csv file
        self.data = pd.read_csv(csv_file)
        
        # fill NaN values with 0.0
        self.data.fillna(0.0, inplace=True)

        # get the report impressions
        self.texts = self.data["Report Impression"].values

        # get the labels
        self.labels = self.data.iloc[:, 1:15].values.astype(float)

    def __len__(self):
        # the length of the dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # get the text and label at the given index
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text, label
