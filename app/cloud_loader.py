from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def load_model_from_hf(model_name: str, cache_dir: str = "app/model"):
    """Загружает модель и токенизатор из Hugging Face Hub"""
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")