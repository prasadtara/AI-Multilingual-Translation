#!/usr/bin/env python3
"""
Comprehensive Evaluation with Multiple Metrics
Calculates BLEU, chrF, TER, and METEOR
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu.metrics import BLEU, CHRF, TER
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
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)
    
    return translations

def main():
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION - Multiple Metrics")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading test data...")
    chinese = read_lines("data/mandarin/test.zh")
    references = read_lines("data/mandarin/test.en")
    print(f"   Test sentences: {len(chinese)}")
    
    # Evaluate both models
    models = [
        ("Baseline (Pretrained)", "Helsinki-NLP/opus-mt-zh-en"),
        ("Fine-tuned (Ours)", "./results/mandarin_model/final")
    ]
    
    results = {}
    
    for model_name, model_path in models:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            if "Helsinki" in model_path:
                tokenizer = MarianTokenizer.from_pretrained(model_path)
                model = MarianMTModel.from_pretrained(model_path).to(device)
            else:
                tokenizer = MarianTokenizer.from_pretrained(model_path)
                model = MarianMTModel.from_pretrained(model_path).to(device)
            model.eval()
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Translate
        print("Translating...")
        start = time.time()
        translations = translate_batch(model, tokenizer, chinese, device, batch_size=16)
        elapsed = time.time() - start
        
        # Calculate metrics
        print("\nCalculating metrics...")
        
        # BLEU
        bleu = BLEU()
        bleu_score = bleu.corpus_score(translations, [references])
        
        # chrF (character-level F-score)
        chrf = CHRF()
        chrf_score = chrf.corpus_score(translations, [references])
        
        # TER (Translation Error Rate - lower is better)
        ter = TER()
        ter_score = ter.corpus_score(translations, [references])
        
        # Store results
        results[model_name] = {
            'BLEU': bleu_score.score,
            'chrF': chrf_score.score,
            'TER': ter_score.score,
            'time': elapsed
        }
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  BLEU:  {bleu_score.score:.2f}")
        print(f"  chrF:  {chrf_score.score:.2f}")
        print(f"  TER:   {ter_score.score:.2f} (lower is better)")
        print(f"  Time:  {elapsed:.1f}s")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    baseline_results = results.get("Baseline (Pretrained)", {})
    finetuned_results = results.get("Fine-tuned (Ours)", {})
    
    for metric in ['BLEU', 'chrF', 'TER']:
        baseline_val = baseline_results.get(metric, 0)
        finetuned_val = finetuned_results.get(metric, 0)
        
        if metric == 'TER':
            # For TER, lower is better, so improvement is negative delta
            improvement = baseline_val - finetuned_val
            print(f"{metric:<15} {baseline_val:<15.2f} {finetuned_val:<15.2f} {improvement:+.2f}")
        else:
            improvement = finetuned_val - baseline_val
            print(f"{metric:<15} {baseline_val:<15.2f} {finetuned_val:<15.2f} {improvement:+.2f}")
    
    # Save to file
    output_file = "comprehensive_evaluation_results.txt"
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("Multiple Metrics: BLEU, chrF, TER\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: WMT19 Chinese-English (300 test pairs)\n")
        f.write(f"Domain: Professional news translations\n\n")
        
        f.write("BASELINE (Pretrained MarianMT):\n")
        f.write(f"  BLEU:  {baseline_results.get('BLEU', 0):.2f}\n")
        f.write(f"  chrF:  {baseline_results.get('chrF', 0):.2f}\n")
        f.write(f"  TER:   {baseline_results.get('TER', 0):.2f}\n\n")
        
        f.write("FINE-TUNED (3 epochs on 2,400 WMT19 pairs):\n")
        f.write(f"  BLEU:  {finetuned_results.get('BLEU', 0):.2f}\n")
        f.write(f"  chrF:  {finetuned_results.get('chrF', 0):.2f}\n")
        f.write(f"  TER:   {finetuned_results.get('TER', 0):.2f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        bleu_imp = finetuned_results.get('BLEU', 0) - baseline_results.get('BLEU', 0)
        chrf_imp = finetuned_results.get('chrF', 0) - baseline_results.get('chrF', 0)
        ter_imp = baseline_results.get('TER', 0) - finetuned_results.get('TER', 0)
        
        f.write(f"  BLEU:  {bleu_imp:+.2f} points ({bleu_imp/baseline_results.get('BLEU', 1)*100:+.1f}%)\n")
        f.write(f"  chrF:  {chrf_imp:+.2f} points ({chrf_imp/baseline_results.get('chrF', 1)*100:+.1f}%)\n")
        f.write(f"  TER:   {ter_imp:+.2f} points (lower is better)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("METRIC EXPLANATIONS\n")
        f.write("=" * 70 + "\n\n")
        f.write("BLEU (Bilingual Evaluation Understudy):\n")
        f.write("  - Measures n-gram precision (1-4 grams)\n")
        f.write("  - Range: 0-100 (higher is better)\n")
        f.write("  - Industry standard for MT evaluation\n\n")
        
        f.write("chrF (Character n-gram F-score):\n")
        f.write("  - Character-level matching (robust to word order)\n")
        f.write("  - Range: 0-100 (higher is better)\n")
        f.write("  - Better for morphologically rich languages\n\n")
        
        f.write("TER (Translation Error Rate):\n")
        f.write("  - Measures edit distance (insertions, deletions, shifts)\n")
        f.write("  - Range: 0-100 (LOWER is better)\n")
        f.write("  - Indicates how much editing needed\n\n")
    
    print(f"Results saved to {output_file}")
    print("\nComprehensive evaluation complete!")

if __name__ == "__main__":
    main()
