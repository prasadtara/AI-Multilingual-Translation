# scripts/preprocess_mandarin.py
import random
import os

def load_parallel_data(en_file, zh_file):
    """Load parallel English-Chinese sentence pairs"""
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
    
    # Clean and pair sentences
    pairs = []
    for en, zh in zip(en_lines, zh_lines):
        en = en.strip()
        zh = zh.strip()
        # Filter out very short or very long sentences
        if 5 <= len(en.split()) <= 50 and 5 <= len(zh) <= 100:
            pairs.append((zh, en))  # (source, target)
    
    return pairs

def split_data(pairs, train_ratio=0.8, dev_ratio=0.1):
    """Split data into train/dev/test sets"""
    random.shuffle(pairs)
    
    total = len(pairs)
    train_end = int(total * train_ratio)
    dev_end = int(total * (train_ratio + dev_ratio))
    
    train = pairs[:train_end]
    dev = pairs[train_end:dev_end]
    test = pairs[dev_end:]
    
    return train, dev, test

def save_split(pairs, prefix, output_dir):
    """Save parallel data to files"""
    zh_file = os.path.join(output_dir, f"{prefix}.zh")
    en_file = os.path.join(output_dir, f"{prefix}.en")
    
    with open(zh_file, 'w', encoding='utf-8') as f_zh:
        with open(en_file, 'w', encoding='utf-8') as f_en:
            for zh, en in pairs:
                f_zh.write(zh + '\n')
                f_en.write(en + '\n')
    
    print(f"Saved {len(pairs)} pairs to {prefix}.zh and {prefix}.en")

def main():
    # File paths
    data_dir = "../data"
    en_file = os.path.join(data_dir, "News-Commentary.en-zh_CN.en")
    zh_file = os.path.join(data_dir, "News-Commentary.en-zh_CN.zh_CN")
    
    print("Loading data...")
    pairs = load_parallel_data(en_file, zh_file)
    print(f"Loaded {len(pairs)} sentence pairs")
    
    print("\nSplitting data...")
    train, dev, test = split_data(pairs)
    
    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")
    
    print("\nSaving splits...")
    save_split(train, "train", data_dir)
    save_split(dev, "dev", data_dir)
    save_split(test, "test", data_dir)
    
    print("\nSample pairs:")
    for i in range(3):
        zh, en = train[i]
        print(f"ZH: {zh}")
        print(f"EN: {en}")
        print()

if __name__ == "__main__":
    main()
