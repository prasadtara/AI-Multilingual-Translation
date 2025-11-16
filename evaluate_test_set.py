#!/usr/bin/env python3
"""
Comprehensive evaluation on the full 300-pair test set
Calculates BLEU score using sacrebleu
"""

from transformers import MarianMTModel, MarianTokenizer
import torch
import os

def load_test_data(data_dir="data/mandarin"):
    """Load test set"""
    zh_file = os.path.join(data_dir, "test.zh")
    en_file = os.path.join(data_dir, "test.en")
    
    with open(zh_file, 'r', encoding='utf-8') as f:
        chinese_sentences = [line.strip() for line in f.readlines()]
    
    with open(en_file, 'r', encoding='utf-8') as f:
        english_references = [line.strip() for line in f.readlines()]
    
    return chinese_sentences, english_references

def translate_batch(sentences, model, tokenizer, batch_size=8):
    """Translate a list of sentences in batches"""
    translations = []
    
    print(f"\nTranslating {len(sentences)} sentences...")
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=128,
                num_beams=4,  # Beam search for better quality
                early_stopping=True
            )
        
        # Decode
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)
        
        # Progress indicator
        if (i + batch_size) % 40 == 0 or i + batch_size >= len(sentences):
            print(f"  Progress: {min(i + batch_size, len(sentences))}/{len(sentences)} sentences")
    
    return translations

def calculate_bleu_manual(predictions, references):
    """
    Simple BLEU calculation without sacrebleu dependency
    Note: This is approximate - sacrebleu is more accurate
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def bleu_score(pred, ref):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        scores = []
        for n in range(1, 5):  # 1-gram to 4-gram
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                scores.append(0)
                continue
            
            matches = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
            total = len(pred_ngrams)
            scores.append(matches / total if total > 0 else 0)
        
        # Geometric mean
        if all(s > 0 for s in scores):
            geo_mean = math.exp(sum(math.log(s) for s in scores) / 4)
        else:
            geo_mean = 0
        
        # Brevity penalty
        bp = min(1, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        
        return bp * geo_mean * 100
    
    # Calculate for all pairs
    scores = [bleu_score(pred, ref) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)

def main():
    print("="*70)
    print("BASELINE MODEL EVALUATION - WMT19 TEST SET")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading pretrained MarianMT model...")
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    print(f"Model loaded: {model_name}")
    
    # Load test data
    print("\n[2/4] Loading test set...")
    chinese_sentences, english_references = load_test_data()
    print(f"Loaded {len(chinese_sentences)} test pairs")
    
    # Translate all test sentences
    print("\n[3/4] Translating test set...")
    print("This will take 2-3 minutes...")
    translations = translate_batch(chinese_sentences, model, tokenizer, batch_size=8)
    print("Translation complete")
    
    # Calculate BLEU score
    print("\n[4/4] Calculating BLEU score...")
    
    # Try using sacrebleu if available
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(translations, [english_references])
        bleu_score = bleu.score
        print(f"Using sacrebleu (accurate)")
    except ImportError:
        print("Using approximate BLEU calculation")
        bleu_score = calculate_bleu_manual(translations, english_references)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n BLEU Score: {bleu_score:.2f}")
    
    # Interpretation
    print("\nInterpretation:")
    if bleu_score >= 40:
        print("  Excellent quality (40+)")
    elif bleu_score >= 30:
        print("  Good quality (30-40)")
    elif bleu_score >= 20:
        print("  Decent quality (20-30)")
    else:
        print("  Needs improvement (<20)")
    
    # Show sample translations
    print("\n" + "="*70)
    print("SAMPLE TRANSLATIONS (First 10 from test set)")
    print("="*70)
    
    for i in range(min(10, len(chinese_sentences))):
        print(f"\n[{i+1}]")
        print(f"Chinese:    {chinese_sentences[i]}")
        print(f"Reference:  {english_references[i]}")
        print(f"Prediction: {translations[i]}")
        
        # Simple match check
        if translations[i].lower().strip() == english_references[i].lower().strip():
            print("  Perfect match!")
        elif len(set(translations[i].lower().split()) & set(english_references[i].lower().split())) > len(english_references[i].split()) * 0.5:
            print("  ~ Good overlap")
    
    # Save results
    output_file = "baseline_test_results.txt"
    print(f"\n[5/5] Saving detailed results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE MODEL TEST RESULTS - WMT19\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: WMT19 Chinese-English\n")
        f.write(f"Test set size: {len(chinese_sentences)} pairs\n")
        f.write(f"BLEU Score: {bleu_score:.2f}\n\n")
        f.write("="*70 + "\n")
        f.write("ALL TRANSLATIONS\n")
        f.write("="*70 + "\n\n")
        
        for i, (zh, ref, pred) in enumerate(zip(chinese_sentences, english_references, translations)):
            f.write(f"[{i+1}]\n")
            f.write(f"Chinese:    {zh}\n")
            f.write(f"Reference:  {ref}\n")
            f.write(f"Prediction: {pred}\n\n")
    
    print(f"Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nBaseline BLEU Score: {bleu_score:.2f}")

if __name__ == "__main__":
    main()
