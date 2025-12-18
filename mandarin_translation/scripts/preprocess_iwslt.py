#!/usr/bin/env python3
"""
Preprocess IWSLT Chinese-English dataset
"""

import os
import random

def load_file(filepath):
    """Load lines from a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_parallel_files(zh_lines, en_lines, prefix, output_dir):
    """Save parallel Chinese-English files"""
    zh_file = os.path.join(output_dir, f"{prefix}.zh")
    en_file = os.path.join(output_dir, f"{prefix}.en")
    
    with open(zh_file, 'w', encoding='utf-8') as f_zh:
        with open(en_file, 'w', encoding='utf-8') as f_en:
            for zh, en in zip(zh_lines, en_lines):
                f_zh.write(zh + '\n')
                f_en.write(en + '\n')
    
    print(f"✓ Saved {len(zh_lines)} pairs to {prefix}.zh and {prefix}.en")

def main():
    # Input files (adjust paths based on actual extracted structure)
    input_dir = "data/mandarin/zh-en"
    zh_file = os.path.join(input_dir, "train.tags.zh-en.zh")
    en_file = os.path.join(input_dir, "train.tags.zh-en.en")
    
    output_dir = "data/mandarin"
    
    print("Loading data...")
    zh_lines = load_file(zh_file)
    en_lines = load_file(en_file)
    
    # Pair and shuffle
    pairs = list(zip(zh_lines, en_lines))
    random.seed(42)
    random.shuffle(pairs)
    
    # Split (80/10/10)
    total = len(pairs)
    train_end = int(0.8 * total)
    dev_end = int(0.9 * total)
    
    train_pairs = pairs[:train_end]
    dev_pairs = pairs[train_end:dev_end]
    test_pairs = pairs[dev_end:]
    
    print(f"\nTotal pairs: {total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Dev: {len(dev_pairs)}")
    print(f"Test: {len(test_pairs)}")
    
    # Save
    print("\nSaving splits...")
    save_parallel_files([p[0] for p in train_pairs], [p[1] for p in train_pairs], "train", output_dir)
    save_parallel_files([p[0] for p in dev_pairs], [p[1] for p in dev_pairs], "dev", output_dir)
    save_parallel_files([p[0] for p in test_pairs], [p[1] for p in test_pairs], "test", output_dir)
    
    print("\nSample pairs:")
    for i in range(3):
        print(f"\nZH: {train_pairs[i][0]}")
        print(f"EN: {train_pairs[i][1]}")

if __name__ == "__main__":
    main()
