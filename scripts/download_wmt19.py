#!/usr/bin/env python3
"""
Download WMT19 Chinese-English dataset from Hugging Face
"""

from datasets import load_dataset
import os

def download_and_save_wmt19():
    print("="*70)
    print("DOWNLOADING WMT19 CHINESE-ENGLISH DATASET")
    print("="*70)
    
    # Create data directory if it doesn't exist
    os.makedirs("data/mandarin", exist_ok=True)
    
    print("\n[1/5] Loading WMT19 dataset from Hugging Face...")
    print("This may take 2-3 minutes on first download...")
    
    try:
        # Load WMT19 Chinese-English dataset
        dataset = load_dataset("wmt19", "zh-en", split="train")
        print(f"✓ Loaded {len(dataset)} total examples from WMT19")
        
    except Exception as e:
        print(f"✗ Error loading WMT19: {e}")
        print("\nTrying alternative: News-Commentary corpus...")
        dataset = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train")
        print(f"✓ Loaded {len(dataset)} examples from News-Commentary")
    
    # Select subset
    print("\n[2/5] Selecting 3,000 examples...")
    if len(dataset) > 3000:
        dataset = dataset.select(range(3000))
    else:
        print(f"Warning: Dataset only has {len(dataset)} examples")
    
    print(f"✓ Selected {len(dataset)} examples")
    
    # Split into train/dev/test
    print("\n[3/5] Splitting into train/dev/test...")
    train_size = 2400
    dev_size = 300
    test_size = 300
    
    train_data = dataset.select(range(train_size))
    dev_data = dataset.select(range(train_size, train_size + dev_size))
    test_data = dataset.select(range(train_size + dev_size, train_size + dev_size + test_size))
    
    print(f"✓ Train: {len(train_data)} examples")
    print(f"✓ Dev:   {len(dev_data)} examples")
    print(f"✓ Test:  {len(test_data)} examples")
    
    # Save to files
    print("\n[4/5] Saving to data/mandarin/...")
    
    def save_split(data, prefix):
        zh_file = f"data/mandarin/{prefix}.zh"
        en_file = f"data/mandarin/{prefix}.en"
        
        with open(zh_file, 'w', encoding='utf-8') as f_zh, \
             open(en_file, 'w', encoding='utf-8') as f_en:
            
            for example in data:
                # WMT19 format has 'translation' field with 'zh' and 'en' keys
                if 'translation' in example:
                    zh_text = example['translation']['zh'].strip()
                    en_text = example['translation']['en'].strip()
                elif 'zh' in example and 'en' in example:
                    zh_text = example['zh'].strip()
                    en_text = example['en'].strip()
                else:
                    print(f"Warning: Unknown format: {example.keys()}")
                    continue
                
                # Skip empty lines
                if zh_text and en_text:
                    f_zh.write(zh_text + '\n')
                    f_en.write(en_text + '\n')
        
        print(f"✓ Saved {prefix}.zh and {prefix}.en")
    
    save_split(train_data, "train")
    save_split(dev_data, "dev")
    save_split(test_data, "test")
    
    # Verify
    print("\n[5/5] Verifying files...")
    for split in ['train', 'dev', 'test']:
        zh_file = f"data/mandarin/{split}.zh"
        en_file = f"data/mandarin/{split}.en"
        
        with open(zh_file, 'r', encoding='utf-8') as f:
            zh_count = len(f.readlines())
        with open(en_file, 'r', encoding='utf-8') as f:
            en_count = len(f.readlines())
        
        print(f"✓ {split}: {zh_count} Chinese, {en_count} English")
        
        if zh_count != en_count:
            print(f"  ⚠ WARNING: Line count mismatch!")
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE DATA (First 3 examples from training set)")
    print("="*70)
    
    with open("data/mandarin/train.zh", 'r', encoding='utf-8') as f_zh, \
         open("data/mandarin/train.en", 'r', encoding='utf-8') as f_en:
        
        for i in range(3):
            zh = f_zh.readline().strip()
            en = f_en.readline().strip()
            print(f"\n[{i+1}]")
            print(f"ZH: {zh}")
            print(f"EN: {en}")
    
    print("\n" + "="*70)
    print("✓ WMT19 DATA DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nYou can now use this data for training:")
    print("  data/mandarin/train.zh + train.en (2,400 pairs)")
    print("  data/mandarin/dev.zh + dev.en (300 pairs)")
    print("  data/mandarin/test.zh + test.en (300 pairs)")

if __name__ == "__main__":
    download_and_save_wmt19()
