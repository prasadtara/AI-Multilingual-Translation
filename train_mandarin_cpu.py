#!/usr/bin/env python3
"""
Mandarin to English Translation - CPU VERSION
For when GPUs are unavailable
"""

import os
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

def load_data(data_dir="data/mandarin"):
    """Load parallel Mandarin-English data"""
    def read_parallel_files(prefix):
        zh_file = os.path.join(data_dir, f"{prefix}.zh")
        en_file = os.path.join(data_dir, f"{prefix}.en")
        
        with open(zh_file, 'r', encoding='utf-8') as f_zh:
            zh_lines = [line.strip() for line in f_zh]
        
        with open(en_file, 'r', encoding='utf-8') as f_en:
            en_lines = [line.strip() for line in f_en]
        
        return {'mandarin': zh_lines, 'english': en_lines}
    
    print("Loading datasets...")
    train_data = read_parallel_files("train")
    dev_data = read_parallel_files("dev")
    test_data = read_parallel_files("test")
    
    print(f"✓ Train: {len(train_data['mandarin'])} pairs")
    print(f"✓ Dev: {len(dev_data['mandarin'])} pairs")
    print(f"✓ Test: {len(test_data['mandarin'])} pairs")
    
    return Dataset.from_dict(train_data), Dataset.from_dict(dev_data), Dataset.from_dict(test_data)

def preprocess_function(examples, tokenizer, max_length=64):  # Reduced for CPU
    """Tokenize input-output pairs"""
    inputs = examples["mandarin"]
    targets = examples["english"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("="*60)
    print("Mandarin-English Translation - CPU TRAINING")
    print("="*60)
    
    # Force CPU
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Load datasets
    train_dataset, dev_dataset, test_dataset = load_data()
    
    # Load model and tokenizer
    print("Loading pre-trained MarianMT model...")
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    print(f"✓ Model loaded: {model_name}")
    
    # Use smaller subset for CPU training
    print("Using smaller subset for CPU training...")
    small_train = Dataset.from_dict({
        'mandarin': train_dataset['mandarin'][:100],  # First 100 examples
        'english': train_dataset['english'][:100]
    })
    small_dev = Dataset.from_dict({
        'mandarin': dev_dataset['mandarin'][:50],
        'english': dev_dataset['english'][:50]
    })
    
    # Tokenize
    tokenized_train = small_train.map(
        lambda x: preprocess_function(x, tokenizer, max_length=64),
        batched=True,
        remove_columns=small_train.column_names
    )
    
    tokenized_dev = small_dev.map(
        lambda x: preprocess_function(x, tokenizer, max_length=64),
        batched=True,
        remove_columns=small_dev.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments for CPU
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/mandarin_model_cpu",
        eval_strategy="steps",
        eval_steps=10,
        learning_rate=3e-5,
        per_device_train_batch_size=2,  # Small for CPU
        per_device_eval_batch_size=2,
        num_train_epochs=1,  # Just 1 epoch for demo
        save_total_limit=1,
        predict_with_generate=True,
        fp16=False,
        logging_steps=5,
        report_to="none",
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting CPU training (this will take 10-30 minutes)...")
    trainer.train()
    
    # Quick test
    print("\nTesting sample translations...")
    test_sentences = ["你好世界", "今天天气很好"]
    
    for zh_text in test_sentences:
        inputs = tokenizer(zh_text, return_tensors="pt")
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, max_length=50)
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"ZH: {zh_text} -> EN: {translation}")
    

if __name__ == "__main__":
    main()
