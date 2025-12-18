#!/usr/bin/env python3
"""
Mandarin to English Translation using Transformer (MarianMT)
Training Script
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

def preprocess_function(examples, tokenizer, max_length=128):
    """Tokenize input-output pairs"""
    inputs = examples["mandarin"]
    targets = examples["english"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
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
    print("Mandarin-English Translation Training")
    print("Using Transformer Architecture (MarianMT)")
    print("="*60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected. Training will be slow.")
    
    # Load datasets
    print("\n" + "-"*60)
    train_dataset, dev_dataset, test_dataset = load_data()
    
    # Load model and tokenizer - USING MarianMT instead of M2M100
    print("\n" + "-"*60)
    print("Loading pre-trained MarianMT model...")
    model_name = "Helsinki-NLP/opus-mt-zh-en"  # Specifically for Chinese to English
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Model parameters: {model.num_parameters() / 1e6:.1f}M")
    
    # Tokenize datasets
    print("\n" + "-"*60)
    print("Tokenizing datasets...")
    
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_dev = dev_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names
    )
    
    print("✓ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    
    # Training arguments
    print("\n" + "-"*60)
    print("Setting up training configuration...")
    
    output_dir = "./results/mandarin_model"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,  # Increased for A100
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,  # Disable fp16 to avoid compatibility issues
        logging_dir='./logs',
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
    )
    
    print(f"✓ Batch size: {training_args.per_device_train_batch_size}")
    print(f"✓ Learning rate: {training_args.learning_rate}")
    print(f"✓ Epochs: {training_args.num_train_epochs}")
    print(f"✓ Mixed precision (fp16): {training_args.fp16}")
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,  # Keep this for now, but it's deprecated
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print("(This may take 20-40 minutes)\n")
    
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "-"*60)
    print("Saving model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print(f"✓ Model saved to {output_dir}/final")
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final training loss: {train_result.training_loss:.4f}")
    print(f"Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # Quick test
    print("\n" + "-"*60)
    print("Testing model with sample translations...")
    print("-"*60)
    
    test_sentences = [
        "你好世界",
        "今天天气很好", 
        "我喜欢学习人工智能"
    ]
    
    model.eval()
    for zh_text in test_sentences:
        inputs = tokenizer(zh_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=50,
                num_beams=5
            )
        
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"\nZH: {zh_text}")
        print(f"EN: {translation}")
    
    print("\n" + "="*60)
    print("All done! Your model is ready.")
    print("="*60)

if __name__ == "__main__":
    main()
