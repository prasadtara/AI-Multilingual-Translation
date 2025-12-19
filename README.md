### Overview  
This is a semester project for course CS 4820.  
Our project aims to build a multilingual translator for Mandarin-English and Tamil-English.    
  
### Getting Started  
```bash
pip install -r requirements.txt
```
```bash
pip install pandas torch transformers sacrebleu nltk tqdm numpy requests
```

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
**Scripts:**  
- train_mandarin.py               // fine-tuning script for MarianMT  
- evaluate_finetuned.py           // evaluation script for fine-tuned model  
- comprehensive_evaluation.py     // multi-metric evaluation

**Results:**  
- baseline_test_results.txt
- finetuned_test_results.txt
- comprehensive_evaluation_results.txt
- train_mandarin_38325.out        // training log from successful run
  
**Data:**  
- data/mandarin/                  // WMT19 Chinese-English parallel corpus

**7. View Results**  
```bash
cat finetuned_test_results.txt  
cat comprehensive_evaluation_results.txt
```
### Tamil-English Translation

**1. Go to the Tamil Directory:**  
```bash
cd tamil_translation
```
**2. Ensure Thirukkural dataset is in current directory (file already included)**  

**3. Run translation scripts:**  
```bash
python final_embeddings.py
```
  
Results are saved to **translated_tamil_thirukkural.csv**  

**5. Move translated_tamil_thirukkural.csv to working directory if not already**

**5. Calculate Metrics and View Results**  
```bash
python translation_metrics.py
```
**Results:**
- translated_tamil_thirukkural.csv
