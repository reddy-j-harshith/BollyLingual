import sacrebleu
from rouge_score import rouge_scorer  # ✅ Correct module
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score