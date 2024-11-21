import torch
from transformers import Trainer
from llm import LLM_LoRA, TextDataset
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    config_path = "LLM_LoRA/adapter_config.json"
    dataset_name = 'microsoft/orca-agentinstruct-1M-v1'
    dataset = TextDataset(dataset_name)
    dataset.set_transform()
    model = LLM_LoRA(config_path)
    training_args = model.train()
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=dataset.train_ds,
        eval_dataset=dataset.val_ds,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    print(train_result)

if __name__ == "__main__":
    main()
