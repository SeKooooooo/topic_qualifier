from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch

HF_MODEL_NAME = 'SeKooooo/project'

class TextClassifier:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.labels = ['Бывший СССР','Интернет и СМИ','Культура','Мир','Наука и техника','Россия', 'Спорт', 'Экономика' ]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = torch.argmax(probs).item()
        
        return {
            "label": self.labels[predicted_idx],
            "confidence": round(probs[0][predicted_idx].item(), 4)
        }