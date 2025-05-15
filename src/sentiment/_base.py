from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
SentimentLabel = Literal["positive", "negative", "neutral"]


class SentimentOutput(BaseModel):
    label: SentimentLabel = Field(..., description="Sentiment label")
    confidence: float = Field(..., description="Confidence score (0-1)")


class SentimentClassifierOutput(BaseModel):
    predictions: list[SentimentOutput] = Field(
        ...,
        description="List of sentiment predictions with confidence scores",
    )


class BaseSentimentClassifier(ABC):
    """
    Base class for sentiment classifiers.
    """

    @abstractmethod
    def classify(self, text: str) -> SentimentClassifierOutput:
        """
        Classify the sentiment of the given text.
        """
        pass
