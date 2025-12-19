pip install sacrebleu pandas

import pandas as pd
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu


df = pd.read_csv("translated_tamil_thirukkural.csv")


references = df['explanation'].astype(str).tolist()
hypotheses = df['Marian_English'].astype(str).tolist()


list_of_references_sacrebleu = [references]

list_of_references_nltk = [[ref] for ref in references]


# --- CALCULATE BLEU  ---
# Weights=(1/4, 1/4, 1/4, 1/4)
bleu_score = corpus_bleu(list_of_references_nltk, hypotheses)
print(f"BLEU Score (NLTK): {bleu_score:.4f} (20.02 on 100-point scale)")


# --- CALCULATE chrF  ---
chrf_result = sacrebleu.corpus_chrf(hypotheses, list_of_references_sacrebleu)
chrf_score = chrf_result.score
print(f"chrF Score (sacrebleu, out of 100): {chrf_score:.4f}")


# --- CALCULATE TER ---
ter_result = sacrebleu.corpus_ter(hypotheses, list_of_references_sacrebleu)
ter_score = ter_result.score
print(f"TER Score (sacrebleu, Percentage): {ter_score:.4f}")


