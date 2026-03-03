from typing import Optional
import stanza

_TAG_TO_PLACEHOLDER: dict[str, str] = {
    "<URL>": "URLTAG",
    "<PHONE>": "PHONETAG",
    "<EMAIL>": "EMAILTAG",
}
_PLACEHOLDER_TO_TAG: dict[str, str] = {v: k for k, v in _TAG_TO_PLACEHOLDER.items()}


def _protect_tags(text: str) -> str:
    for tag, placeholder in _TAG_TO_PLACEHOLDER.items():
        text = text.replace(tag, placeholder)
    return text


def _restore_tags_in_token(token: str) -> str:
    return _PLACEHOLDER_TO_TAG.get(token, token)

def init_stanza(lang: str = "uk", use_gpu: bool = False) -> stanza.Pipeline:
    stanza.download(lang, verbose=False)
    return stanza.Pipeline(
        lang=lang,
        processors="tokenize,pos,lemma",
        use_gpu=use_gpu,
        verbose=False,
    )

def extract_lemma_pos(doc: stanza.Document) -> dict:
    tokens: list[str] = []
    lemmas: list[str] = []
    pos_tags: list[str] = []

    for sent in doc.sentences:
        for word in sent.words:
            upos = word.upos or "X"

            if upos == "PUNCT":
                continue

            raw_token = _restore_tags_in_token(word.text)
            raw_lemma = word.lemma if (word.lemma and word.lemma != "_") else word.text
            raw_lemma = _restore_tags_in_token(raw_lemma)

            tokens.append(raw_token)
            lemmas.append(raw_lemma)
            pos_tags.append(upos)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "lemma_text": " ".join(lemmas),
        "upos_seq": " ".join(pos_tags),
    }


def process_text(text: str, nlp: stanza.Pipeline) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "tokens": [],
            "lemmas": [],
            "pos_tags": [],
            "lemma_text": "",
            "upos_seq": "",
        }

    protected = _protect_tags(text)
    doc = nlp(protected)
    return extract_lemma_pos(doc)

def filter_by_pos(
    lemmas: list[str],
    pos_tags: list[str],
    keep: tuple[str, ...] = ("NOUN", "ADJ", "PROPN"),
) -> str:
    filtered = [l for l, p in zip(lemmas, pos_tags) if p in keep]
    return " ".join(filtered)

def batch_process_df(
    df,
    nlp: stanza.Pipeline,
    text_col: str = "text_v2",
    batch_size: int = 32,
    save_every: int = 500,
    checkpoint_path: Optional[str] = None,
) -> list[dict]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **_): 
            return iterable

    texts = df[text_col].fillna("").tolist()
    results: list[dict] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Stanza lemmatisation"):
        batch_texts = texts[i : i + batch_size]
        for text in batch_texts:
            results.append(process_text(text, nlp))

        if checkpoint_path and len(results) % save_every < batch_size:
            import pandas as pd

            partial = df.iloc[: len(results)].copy()
            partial["lemma_text"] = [r["lemma_text"] for r in results]
            partial["upos_seq"] = [r["upos_seq"] for r in results]
            partial.to_csv(checkpoint_path, index=False)

    return results
