import transformers
import accelerate
import peft
from vit import Dataset, ViT_LoRA
import utils
from transformers import Trainer
import evaluate
import numpy as np

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")



dataset = Dataset()
model = ViT_LoRA(dataset)
dataset.set_transform()
model.lora()

train_args = model.train()

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    # assert False
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    accuracy = metric.compute(predictions=predictions, references=references)
    print(accuracy)
    return {"eval_accuracy": accuracy["accuracy"]}

trainer = Trainer(
    model.model,
    args=train_args,
    train_dataset=dataset.train_ds,
    eval_dataset=dataset.val_ds,
    processing_class=model.image_processor,
    compute_metrics=compute_metrics,
    data_collator=utils.collate_fn,
)
# print(trainer.evaluate(dataset.val_ds))
train_results = trainer.train()