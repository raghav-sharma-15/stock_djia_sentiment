# src/sentiment_extractor.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class FinBERTSentiment:
    def __init__(self, device: str = None):
        """
        Load FinBERT (ProsusAI/finbert) model & tokenizer.
        Uses GPU if available; else CPU. Force safetensors loading via use_safetensors=True.
        """
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Tell Transformers to load the safetensors file if available,
        # avoiding any torch.load on a .bin that triggers the vulnerability check.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            use_safetensors=True
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_sentiment_scores(self, text: str) -> dict:
        """
        Returns a dict with keys {"positive", "negative", "neutral"} and their probabilities.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # ProsusAI/finbert label order: [positive, negative, neutral]
        return {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2])
        }


def attach_sentiment_to_news(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Given a DataFrame with columns ['date', 'label', 'headline'],
    runs FinBERT on each 'headline' (in batches), and returns a new DataFrame
    with added columns ['pos', 'neg', 'neu'] per row.
    """
    finbert = FinBERTSentiment()
    records = []

    for i in tqdm(range(0, len(df), batch_size), desc="FinBERT sentiment"):
        batch = df.iloc[i : i + batch_size]
        texts = batch["headline"].tolist()
        encodings = finbert.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(finbert.device)

        with torch.no_grad():
            outputs = finbert.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for idx_in_batch, (pos, neg, neu) in enumerate(probs):
            row = batch.iloc[idx_in_batch].to_dict()
            row.update({
                "pos": float(pos),
                "neg": float(neg),
                "neu": float(neu)
            })
            records.append(row)

    df_sent = pd.DataFrame(records)
    df_sent["date"] = pd.to_datetime(df_sent["date"])
    return df_sent  # columns: ['date', 'label', 'headline', 'pos', 'neg', 'neu']