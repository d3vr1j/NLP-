#RoBERTa

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

text = "This is a great movie!"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

logits = outputs.logits
print(logits)
