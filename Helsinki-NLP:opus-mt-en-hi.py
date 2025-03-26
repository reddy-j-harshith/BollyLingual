from transformers import pipeline ,AutoTokenizer, AutoModelForSeq2SeqLM
import torch,sacrebleu,nltk
from rouge_score import rouge_scorer
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from datasets import load_dataset
import random


dataset = load_dataset("cfilt/iitb-english-hindi")
test_data = dataset["test"]

def evaluate_translation(reference, hypothesis):
    # Tokenize the reference and hypothesis before passing to METEOR
    reference_tokens = reference.split()  # Tokenizing reference
    hypothesis_tokens = hypothesis.split()  # Tokenizing hypothesis

    # BLEU Score
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, hypothesis)["rougeL"].fmeasure

    # METEOR Score (Now using tokenized input)
    meteor = meteor_score([reference_tokens], hypothesis_tokens)

    return {"BLEU": bleu, "ROUGE": rouge, "METEOR": meteor}

# Define an evaluation function
def evaluate_model(dataset):
    bleu_scores, rouge_scores, meteor_scores = [], [], []
    total_samples = len(dataset["test"])
    print(f"Evaluating on {total_samples} test samples...")
    for i in range(total_samples):
        english = dataset["test"][i]["translation"]["en"]
        reference_hindi = dataset["test"][i]["translation"]["hi"]
        predicted_hindi = preprocess_text(english)

        scores = evaluate_translation(reference_hindi, predicted_hindi)

        bleu_scores.append(scores["BLEU"])
        rouge_scores.append(scores["ROUGE"])
        meteor_scores.append(scores["METEOR"])

        if i % 100 == 0:
            print(f"Processed {i}/{total_samples} sentences...")

    
    avg_bleu = sum(bleu_scores) / total_samples
    avg_rouge = sum(rouge_scores) / total_samples
    avg_meteor = sum(meteor_scores) / total_samples

    return {"BLEU": avg_bleu, "ROUGE": avg_rouge, "METEOR": avg_meteor}


def preprocess_text(text):
    inputs = tokenizer(text,return_tensors = "pt",padding = True,truncation = True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(outputs,skip_special_tokens = True)
    return translated_text[0]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
 
#english_sentence = "Hello, how are you?"
#hindi_translation = preprocess_text(english_sentence)
#print("Translated:", hindi_translation)


#sample = random.choice(test_data)
#english_text = sample["translation"]["en"]
#hindi_reference = sample["translation"]["hi"]
#translated_hindi = preprocess_text(english_text)

#print("Translated Hindi:    ", translated_hindi)
#print("English:     ", english_text)
#print("Reference Hindi:     ", hindi_reference)


scores = evaluate_model(dataset)
print("\nEvaluation Scores:", scores)