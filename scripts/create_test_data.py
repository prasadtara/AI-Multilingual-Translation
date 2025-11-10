#!/usr/bin/env python3
"""
Create a test dataset for Mandarin-English translation
This creates a small but realistic dataset for development and testing
"""

import os
import random

# Curated Mandarin-English parallel sentences
parallel_data = [
    # Greetings and basic phrases
    ("你好", "Hello"),
    ("再见", "Goodbye"),
    ("谢谢你", "Thank you"),
    ("不客气", "You're welcome"),
    ("对不起", "I'm sorry"),
    ("没关系", "It's okay"),
    ("早上好", "Good morning"),
    ("晚安", "Good night"),
    
    # Weather and time
    ("今天天气很好", "The weather is nice today"),
    ("明天会下雨吗", "Will it rain tomorrow"),
    ("现在几点了", "What time is it now"),
    ("今天是星期几", "What day is today"),
    ("天气很冷", "The weather is cold"),
    ("外面很热", "It's hot outside"),
    
    # Daily life
    ("我喜欢学习人工智能", "I like studying artificial intelligence"),
    ("这是一个测试句子", "This is a test sentence"),
    ("我在学校学习", "I study at school"),
    ("我喜欢看书", "I like reading books"),
    ("我每天都锻炼", "I exercise every day"),
    ("我喜欢旅行", "I like traveling"),
    
    # Technology and AI
    ("机器翻译很有趣", "Machine translation is interesting"),
    ("深度学习需要大量数据", "Deep learning requires a lot of data"),
    ("自然语言处理是一个挑战", "Natural language processing is a challenge"),
    ("神经网络可以学习复杂模式", "Neural networks can learn complex patterns"),
    ("这个模型需要训练", "This model needs training"),
    ("翻译质量正在提高", "Translation quality is improving"),
    ("人工智能正在改变世界", "Artificial intelligence is changing the world"),
    ("计算机可以理解语言", "Computers can understand language"),
    
    # Education
    ("我在大学学习计算机科学", "I study computer science at university"),
    ("这门课很有挑战性", "This course is challenging"),
    ("我需要完成我的项目", "I need to complete my project"),
    ("教授很有帮助", "The professor is helpful"),
    ("考试很难", "The exam is difficult"),
    ("我喜欢这个班级", "I like this class"),
    
    # Food and dining
    ("我喜欢中国菜", "I like Chinese food"),
    ("这个很好吃", "This is delicious"),
    ("我饿了", "I'm hungry"),
    ("我想喝水", "I want to drink water"),
    ("请给我菜单", "Please give me the menu"),
    
    # Common verbs and actions
    ("我去商店", "I go to the store"),
    ("他在工作", "He is working"),
    ("她在读书", "She is reading"),
    ("我们在学习", "We are studying"),
    ("他们在说话", "They are talking"),
    ("我正在写作业", "I am doing homework"),
    
    # Questions
    ("你叫什么名字", "What is your name"),
    ("你好吗", "How are you"),
    ("你从哪里来", "Where are you from"),
    ("你会说英语吗", "Do you speak English"),
    ("这是什么", "What is this"),
    ("多少钱", "How much does it cost"),
]

def create_dataset(output_dir="data/mandarin", total_samples=3000):
    """
    Create train/dev/test splits with augmented data
    """
    print("Creating Mandarin-English test dataset...")
    print(f"Target: {total_samples} total samples\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Augment data by repeating and shuffling
    augmented_data = parallel_data * (total_samples // len(parallel_data) + 1)
    augmented_data = augmented_data[:total_samples]
    
    # Shuffle
    random.seed(42)
    random.shuffle(augmented_data)
    
    # Split into train/dev/test (80/10/10)
    train_size = int(0.8 * len(augmented_data))
    dev_size = int(0.1 * len(augmented_data))
    
    train_data = augmented_data[:train_size]
    dev_data = augmented_data[train_size:train_size + dev_size]
    test_data = augmented_data[train_size + dev_size:]
    
    # Save function
    def save_split(data, prefix):
        zh_file = os.path.join(output_dir, f"{prefix}.zh")
        en_file = os.path.join(output_dir, f"{prefix}.en")
        
        with open(zh_file, 'w', encoding='utf-8') as f_zh:
            with open(en_file, 'w', encoding='utf-8') as f_en:
                for zh, en in data:
                    f_zh.write(zh + '\n')
                    f_en.write(en + '\n')
        
        print(f"✓ Saved {len(data)} pairs to {prefix}.zh and {prefix}.en")
    
    # Save all splits
    save_split(train_data, "train")
    save_split(dev_data, "dev")
    save_split(test_data, "test")
    
    print("\n" + "="*60)
    print("SUCCESS! Dataset created")
    print("="*60)
    print(f"Train: {len(train_data)} pairs")
    print(f"Dev: {len(dev_data)} pairs")
    print(f"Test: {len(test_data)} pairs")
    print(f"Total: {len(augmented_data)} pairs")
    
    print("\nSample translations from training set:")
    print("-"*60)
    for i in range(5):
        zh, en = train_data[i]
        print(f"ZH: {zh}")
        print(f"EN: {en}")
        print()
    
    print("Files created in:", output_dir)
    print("Ready for training!")

if __name__ == "__main__":
    create_dataset()
