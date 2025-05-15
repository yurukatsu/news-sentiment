import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ._base import (
    BaseSentimentClassifier,
    SentimentClassifierOutput,
    SentimentOutput,
    SentimentLabel,
)


class HuggingFaceSentimentClassifier(BaseSentimentClassifier):
    """
    A sentiment classifier using Hugging Face Transformers.
    """

    model_name: str
    hf_label2sentiment: dict[str, SentimentLabel]

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify(self, text: str) -> SentimentClassifierOutput:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = logits.softmax(dim=-1).squeeze().tolist()
        labels = self.model.config.id2label
        predictions = [
            SentimentOutput(
                label=self.hf_label2sentiment.get(labels[i], labels[i]),
                confidence=prob,
            )
            for i, prob in enumerate(probabilities)
        ]
        return SentimentClassifierOutput(predictions=predictions)


class FinBERTClassifier(HuggingFaceSentimentClassifier):
    model_name = "ProsusAI/finbert"
    hf_label2sentiment = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }

    def __init__(self):
        super().__init__(model_name=self.model_name)
