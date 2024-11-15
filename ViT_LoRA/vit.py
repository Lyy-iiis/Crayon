import torch
import torch.nn as nn
import transformers
import accelerate
import peft
from transformers import AutoImageProcessor
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments

from peft import LoraConfig, get_peft_model


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import numpy as np
import evaluate


class Dataset:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("food101", split="train[:5000]", cache_dir="./data")
        labels = self.dataset.features["label"].names
        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(labels):
            self.label2id[label] = i
            self.id2label[i] = label
        self.splits()
        
        self.train_transforms = None
        self.val_transforms = None
    
    def splits(self):
        splits = self.dataset.train_test_split(test_size=0.1)
        self.train_ds = splits["train"]
        self.val_ds = splits["test"]
    
    def set_transform(self):
                
        def preprocess_train(example_batch):
            """Apply train_transforms across a batch."""
            example_batch["pixel_values"] = [self.train_transforms(image.convert("RGB")) for image in example_batch["image"]]
            return example_batch


        def preprocess_val(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [self.val_transforms(image.convert("RGB")) for image in example_batch["image"]]
            return example_batch
        
        self.train_ds.set_transform(preprocess_train)
        self.val_ds.set_transform(preprocess_val)
        
class ViT_LoRA(nn.Module):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_checkpoint = "google/vit-base-patch16-224-in21k"
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
        
        self.model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.model_name = model_checkpoint.split("/")[-1]
        self.dataset = dataset
        self.preprare_transform()
    
    def preprare_transform(self):
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        
        self.dataset.train_transforms = Compose(
            [
                RandomResizedCrop(self.image_processor.size["height"]),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

        self.dataset.val_transforms = Compose(
            [
                Resize(self.image_processor.size["height"]),
                CenterCrop(self.image_processor.size["height"]),
                ToTensor(),
                normalize,
            ]
        )

    
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
    
    def lora(self):
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model = get_peft_model(self.model, config)
        self.print_trainable_parameters()
        
    def train(self):
        batch_size = 128
        
        args = TrainingArguments(
            f"{self.model_name}-finetuned-lora-food101",
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
    
    # def eval(self):
    #     metric = evaluate.load("accuracy")
        
    #     def compute_metrics(eval_pred):
    #         """Computes accuracy on a batch of predictions"""
    #         predictions = np.argmax(eval_pred.predictions, axis=1)
    #         return metric.compute(predictions=predictions, references=eval_pred.label_ids)
        
    #     return compute_metrics