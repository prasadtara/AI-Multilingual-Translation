#!/usr/bin/env python3
"""
Download Mandarin-English parallel corpus from Hugging Face
"""

from datasets import load_dataset
import os

def download_and_save():
    print("Downloading Mandarin-English dataset from Hugging Face...")
    print("This may take a few minutes...\n")
    
    # Load WMT19 Chinese-English dataset (high quality)
    dataset = load_dataset("wmt19", "zh-en", split="train")
    
    print(f"Downloaded {len(dataset)} translation pairs!")
    
    # Take a subset (full dataset is huge - 25M pairs)
    # For your project, 50K-100K pairs is plenty
    dataset = dataset.select(range(min(100000, len(dataset))))
    
    print(f"Using {len(dataset)} pairs for training\n")
    
    # Split into train/dev/test (80/10/10)
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))
    
    train_data = dataset.select(range(train_size))
    dev_data = dataset.select(range(train_size, train_size + dev_size))
    test_data = dataset.select(range(train_size + dev_size, len(dataset)))
    
    # Create output directory
    output_dir = "data/mandarin"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to files
    print("Saving files...")
    
    def save_split(data, prefix):
        zh_file = f"{output_dir}/{prefix}.zh"
        en_file = f"{output_dir}/{prefix}.en"
        
        with open(zh_file, 'w', encoding='utf-8') as f_zh:
            with open(en_file, 'w', encoding='utf-8') as f_en:
                for item in data:
                    zh_text = item['translation']['zh']
                    en_text = item['translation']['en']
                    f_zh.write(zh_text + '\n')
                    f_en.write(en_text + '\n')
        
        print(f"✓ Saved {len(data)} pairs to {prefix}.zh and {prefix}.en")
    
    save_split(train_data, "train")
    save_split(dev_data, "dev")
    save_split(test_data, "test")
    
    print("\n" + "="*60)
    print("SUCCESS! Dataset ready for training")
    print("="*60)
    print(f"Train: {len(train_data)} pairs")
    print(f"Dev: {len(dev_data)} pairs")
    print(f"Test: {len(test_data)} pairs")
    print("\nSample translations:")
    for i in range(3):
        print(f"\nZH: {train_data[i]['translation']['zh']}")
        print(f"EN: {train_data[i]['translation']['en']}")

if __name__ == "__main__":
    download_and_save()
