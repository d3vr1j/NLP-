#BERT

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pretrained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

text = "This is a great movie!"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

logits = outputs.logits
print(logits)
