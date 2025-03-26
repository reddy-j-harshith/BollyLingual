from transformers import pipeline ,AutoTokenizer, AutoModelForSeq2SeqLM
import torch,sacrebleu,nltk
from rouge_score import rouge_scorer
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from datasets import load_dataset