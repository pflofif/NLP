import re
from tokenize_uk import tokenize_words, tokenize_sents

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"['`‘’]", "’", text)
    return text

def mask_pii(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    text = re.sub(r'\+?\d{10,12}', '<PHONE>', text)
    return text

def sentence_split(text: str) -> list[str]:
    return tokenize_sents(text)

def preprocess(text: str) -> dict:
    step1 = clean_text(text)
    step2 = mask_pii(step1)
    step3 = normalize_text(step2)
    
    sentences = sentence_split(step3)
    
    final_sentences = []
    for sent in sentences:
        tokens = tokenize_words(sent)
        clean_tokens = [
            t for t in tokens 
            if re.match(r'^[a-zа-яіїєґ0-9’]+$', t) or t in ['<URL>', '<PHONE>', '<EMAIL>']
        ]
        if clean_tokens:
            final_sentences.append(" ".join(clean_tokens))
            
    return {
        "original": text,
        "clean_full": " ".join(final_sentences),
        "sentences": final_sentences
    }