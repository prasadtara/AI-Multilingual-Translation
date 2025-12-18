This is a semester project for course CS 4820.  
Our project aim to build a multilingual translator for Mandarin-English and Tamil-English.    
  
### Getting Started  
``bash
pip install -r requirements.txt
``

**Mandarin-English Translation**  
Go to the Mandarin Directory:  
``bash
cd mandarin_translation
``  

Download WMT19 Dataset (optional, already included):  
``bash
python scripts/download_wmt19.py
``

Evaluate Fine-tuned Model  
``bash
python evaluate_finetuned.py
``   
