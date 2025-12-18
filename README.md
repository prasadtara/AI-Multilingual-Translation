This is a semester project for course CS 4820.  
Our project aim to build a multilingual translator for Mandarin-English and Tamil-English.    
  
### Getting Started  
``bash
pip install -r requirements.txt
``

### Mandarin-English Translation  
**1. Go to the Mandarin Directory:**  
```bash
cd mandarin_translation
```  

**2. Download WMT19 Dataset (optional, already included):**  
```bash
python scripts/download_wmt19.py
```

**3. Evaluate Fine-tuned Model:**  
```bash
python evaluate_finetuned.py
```   
This file translates 300 test sentences and calculates BLEU score.  
Results are saved to **finetuned_test_results.txt**  

**4. Run Comprehensive Evaluation**  
```bash
python comprehensive_evaluation.py
```  
This file evaluates both baseline and fine-tuned models with multiple metrics.  
Metrics: BLEU, chrF, and TER  
Results are saved to **comprehensive_evaluation_results.txt**  

**5. Training (GPU Required)**  
```bash
python train_mandarin.py  
sbatch train_mandarin.sh
```  

**6. Key Files**  
**- Scripts:**  
  train_mandarin.py //fine-tuning script for MarianMT  
