#!/usr/bin/env python3
from transformers import MarianMTModel, MarianTokenizer
import torch

def evaluate_pretrained_model():
    """Evaluate the original pretrained model"""
    print("=== PRETRAINED MODEL EVALUATION ===")
    
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    test_sentences = [
        "你好世界",
        "今天天气很好", 
        "我喜欢学习人工智能",
        "这是一个测试",
        "明天会下雨吗",
        "人工智能很重要"
    ]
    
    print("Pretrained model translations:\n")
    for zh_text in test_sentences:
        inputs = tokenizer(zh_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"'{zh_text}' → '{translation}'")
    
    return model, tokenizer

if __name__ == "__main__":
    evaluate_pretrained_model()
