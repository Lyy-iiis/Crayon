import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import bert_tokenizer
from torch.utils.data import Dataset, DataLoader

class UnlabeledDataset(Dataset):
	"""The dataset to contain report impressions without any labels."""
	
	def __init__(self, csv_path):
		""" Initialize the dataset object
		@param csv_path (string): path to the csv file containing rhe reports. It
									should have a column named "Report Impression"
		
		Example:: (Havn't carefully check for correctness)
			Report Impression
			"There is a small left lower lobe consolidation, likely due to infection."
			"The heart size is within normal limits. No acute pulmonary process."
			
			>>> impressions = [
			>>>         "There is a small left lower lobe consolidation, likely due to infection.",
			>>>         "The heart size is within normal limits. No acute pulmonary process."
			>>> ]
			>>> self.encoded_imp = [
			>>>         [101, 2045, 2003, 1037, 2235, 2187, 2897, 1050, 1005, 1056, 1010, 3497, 2349, 2000, 5657, 1012, 102],
			>>>         [101, 1996, 2540, 2946, 2003, 2306, 3671, 1012, 2053, 11365, 13129, 2833, 1012, 102]
			>>> ]
				
		"""
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		impressions = bert_tokenizer.get_impressions_from_csv(csv_path)
		self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

	def __len__(self):
		"""Compute the length of the dataset

		@return (int): size of the dataframe
		"""
		return len(self.encoded_imp)

	def __getitem__(self, idx):
		""" Functionality to index into the dataset
		@param idx (int): Integer index into the dataset

		@return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
							a LongTensor of an encoded impression. The value of 'label'
							is a LongTensor containing the labels and the value of
							'len' is an integer representing the length of imp's value
		"""
		if torch.is_tensor(idx): # Strange!
				idx = idx.tolist()  # It seems that `idx` can only be a tensor with one element.
		imp = self.encoded_imp[idx]
		imp = torch.LongTensor(imp)
		return {"imp": imp, "len": imp.shape[0]}
