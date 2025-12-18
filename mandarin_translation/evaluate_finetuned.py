#!/usr/bin/env python3
"""
Evaluate Fine-Tuned Mandarin-English Translation Model
Compares fine-tuned model against baseline
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu.metrics import BLEU
import time

def read_lines(filepath):
    """Read lines from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def translate_batch(model, tokenizer, sentences, device, batch_size=16):
    """Translate sentences in batches"""
    translations = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        
        # Decode
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)
    
    return translations

def main():
    print("=" * 60)
    print("Fine-Tuned Model Evaluation")
    print("=" * 60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    model_path = "./results/mandarin_model/final"
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path).to(device)
        model.eval()
        print(" Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"! Error loading model: {e}")
        print("\nTrying to load from Helsinki-NLP checkpoint...")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        model = MarianMTModel.from_pretrained(model_path).to(device)
        model.eval()
    
    # Load test data
    print("\nLoading test data...")
    chinese = read_lines("data/mandarin/test.zh")
    references = read_lines("data/mandarin/test.en")
    
    print(f"   Test sentences: {len(chinese)}")
    
    # Translate
    print("\nTranslating test set...")
    print("   (This will take 2-3 minutes)")
    start_time = time.time()
    
    translations = translate_batch(model, tokenizer, chinese, device, batch_size=16)
    
    elapsed = time.time() - start_time
    print(f"Translation complete in {elapsed:.1f} seconds")
    
    # Calculate BLEU
    print("\nCalculating BLEU score...")
    bleu = BLEU()
    score = bleu.corpus_score(translations, [references])
    
    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL RESULTS")
    print("=" * 60)
    print(f"BLEU Score: {score.score:.2f}")
    print(f"")
    print(f"Baseline BLEU (from baseline_test_results.txt): 23.34")
    print(f"Fine-tuned BLEU: {score.score:.2f}")
    print(f"Improvement: {score.score - 23.34:+.2f} points")
    print("=" * 60)
    
    # Show sample translations
    print("\nSample Translations (first 5):")
    print("-" * 60)
    for i in range(min(5, len(chinese))):
        print(f"\n[{i+1}] Chinese: {chinese[i]}")
        print(f"    Reference: {references[i]}")
        print(f"    Fine-tuned: {translations[i]}")
    
    # Save results
    output_file = "finetuned_test_results.txt"
    print(f"\nSaving detailed results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FINE-TUNED MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Fine-tuned MarianMT (Helsinki-NLP/opus-mt-zh-en)\n")
        f.write(f"Training: 3 epochs on 2,400 WMT19 Chinese-English pairs\n")
        f.write(f"Test set: 300 WMT19 news translations\n\n")
        f.write(f"BLEU Score: {score.score:.2f}\n")
        f.write(f"Baseline BLEU: 23.34\n")
        f.write(f"Improvement: {score.score - 23.34:+.2f} points\n")
        f.write(f"Translation time: {elapsed:.1f} seconds\n\n")
        f.write("=" * 60 + "\n")
        f.write("SAMPLE TRANSLATIONS (First 10)\n")
        f.write("=" * 60 + "\n\n")
        
        for i in range(min(10, len(chinese))):
            f.write(f"[{i+1}]\n")
            f.write(f"Chinese:    {chinese[i]}\n")
            f.write(f"Reference:  {references[i]}\n")
            f.write(f"Fine-tuned: {translations[i]}\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("ALL TRANSLATIONS\n")
        f.write("=" * 60 + "\n\n")
        
        for i in range(len(chinese)):
            f.write(f"[{i+1}]\n")
            f.write(f"ZH: {chinese[i]}\n")
            f.write(f"REF: {references[i]}\n")
            f.write(f"PRED: {translations[i]}\n\n")
    
    print(f"\nResults saved to {output_file}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
