import torch
import torch.nn as nn
import transformers
import accelerate
import peft
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
import json
from datasets import load_dataset
from transformers import AutoTokenizer

class LLM_LoRA(nn.Module):
    def __init__(self, config_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_checkpoint = config["base_model_name_or_path"]
        self.model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
        self.config = config
        self.apply_lora()

    def apply_lora(self):
        lora_config = LoraConfig(
            r=self.config["r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias=self.config["bias"],
            modules_to_save=self.config["modules_to_save"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def train(self):
        batch_size = 128
        args = TrainingArguments(
            f"{self.model_checkpoint.split('/')[-1]}-finetuned-lora",
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-3,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            num_train_epochs=5,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # push_to_hub=True,
            label_names=["labels"],
        )
        
        return args

class TextDataset:
    def __init__(self, dataset_name: str = 'PleIAs/common_corpus', split: str = "train", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset(dataset_name, split=split, cache_dir="./data")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.splits()
        
        self.train_transforms = None
        self.val_transforms = None
    
    def splits(self, test_size: float = 0.1):
        splits = self.dataset.train_test_split(test_size=test_size)
        self.train_ds = splits["train"]
        self.val_ds = splits["test"]
    
    def set_transform(self):
        def preprocess_train(example_batch):
            """Apply train_transforms across a batch."""
            example_batch["input_ids"] = self.tokenizer(example_batch["text"], truncation=True, padding="max_length", max_length=512)["input_ids"]
            example_batch["labels"] = example_batch["input_ids"].copy()
            return example_batch

        def preprocess_val(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["input_ids"] = self.tokenizer(example_batch["text"], truncation=True, padding="max_length", max_length=512)["input_ids"]
            example_batch["labels"] = example_batch["input_ids"].copy()
            return example_batch
        
        self.train_ds = self.train_ds.map(preprocess_train, batched=True)
        self.val_ds = self.val_ds.map(preprocess_val, batched=True)
