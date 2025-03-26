from transformers import pipeline ,AutoTokenizer, AutoModelForSeq2SeqLM
import torch,sacrebleu,nltk
from rouge_score import rouge_scorer
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score


def evaluate_translation(reference, hypothesis):
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, hypothesis)['rougeL'].fmeasure

    meteor = meteor_score([reference], hypothesis)

    return {"BLEU": bleu, "ROUGE": rouge, "METEOR": meteor}


def preprocess_text(text):
    inputs = tokenizer(text,return_tensors = "pt",padding = True,truncation = True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(outputs,skip_special_tokens = True)
    return translated_text[0]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
 
english_sentence = "Hello, how are you?"
hindi_translation = preprocess_text(english_sentence)
print("Translated:", hindi_translation)