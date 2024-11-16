import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm
import argparse

def get_impressions_from_csv(path):	
	df = pd.read_csv(path)
	imp = df['Report Impression']
	imp = imp.fillna('')  # replace NaN with empty string [BUG: if some Report Impression is Nan, it will interperted as a number, which will cause an error `TypeError: expected string or bytes-like object, got 'float'`]
	imp.astype(str)  # convert to string
	imp = imp.str.strip()  # remove leading and trailing whitespaces
	imp = imp.replace('\n',' ', regex=True)  # replace newline characters with spaces
	imp = imp.replace('\s+', ' ', regex=True)  # replace multiple spaces with a single space
	imp = imp.str.strip()  # remove leading and trailing whitespaces
	return imp

def tokenize(impressions, tokenizer):
	new_impressions = []
	print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
	for i in tqdm(range(impressions.shape[0])):
		tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
		"""Example: 
  		>>> impressions.iloc[i] = 'The patient has a mass in the lung.'
  		>>> tokenized_imp = ['the', 'patient', 'has', 'a', 'mass', 'in', 'the', 'lung', '.']
		"""
		if tokenized_imp: #not an empty report
			res = tokenizer.encode_plus(tokenized_imp)['input_ids']  # padding=False and truncation=True are set to default values
			"""Example:
			>>> text = "This is a test."
			>>> encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=10, padding='max_length', truncation=True)
			>>> encoded = {'input_ids': [101, 2023, 2003, 1037, 3231, 1012, 102, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]}
			>>> encoded['input_ids'] = [101, 2023, 2003, 1037, 3231, 1012, 102, 0, 0, 0]
			where 101 is the token id for [CLS], 102 is the token id for [SEP], and 0 is the token id for [PAD]
			"""
			if len(res) > 512: #length exceeds maximum size
				#print("report length bigger than 512")
				res = res[:511] + [tokenizer.sep_token_id]
			new_impressions.append(res)
		else: #an empty report
				new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
	return new_impressions

def load_list(path):
	with open(path, 'r') as filehandle:
		impressions = json.load(filehandle)
		return impressions

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tokenize radiology report impressions and save as a list.')
	parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
						help='path to csv containing reports. The reports should be \
						under the \"Report Impression\" column')
	parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
						help='path to intended output file')
	args = parser.parse_args()
	csv_path = args.data
	out_path = args.output_path
	
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	impressions = get_impressions_from_csv(csv_path)
	new_impressions = tokenize(impressions, tokenizer)
	with open(out_path, 'w') as filehandle:
		json.dump(new_impressions, filehandle)
