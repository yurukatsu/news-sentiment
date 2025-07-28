from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, RootModel, Field

FX_LABELS = ["UP", "DOWN", "FLAT", "MIXED", "NO_SIGNAL"]
FXLabel = Literal["UP", "DOWN", "FLAT", "MIXED", "NO_SIGNAL"]


class FXClassificationScore(BaseModel):
    label: FXLabel = Field(
        ...,
        description="The sentiment label for the FX classification.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification, between 0 and 1.",
    )

class FXClassificationResult(BaseModel):
    asset: str = Field(
        ...,
        description="The asset being classified, e.g., 'USDJPY'.",
    )
    article: str = Field(
        ...,
        description="The article text that was classified.",
    )
    scores: list[FXClassificationScore] = Field(
        ...,
        description="List of FX predictions with confidence scores",
    )
    final_label: FXLabel = Field(
        ...,
        description="Final label determined by the classifier based on the scores.",
    )
    evidence_sentences: list[str] = Field(
        ...,
        description="List of sentences from the input text that support the classification.",
    )
    notes_for_humans: str = Field(
        ...,
        description="Additional notes for human reviewers to understand the classification.",
    )

class FXClassificationOutput(BaseModel):
    results: list[FXClassificationResult] = Field(
        ...,
        description="List of FX classification outputs for each asset.",
    )

class BaseFXClassifier(ABC):
    """
    Base class for FX classifiers.
    """

    @abstractmethod
    def classify(self, texts: list[str]) -> list[FXClassificationOutput]:
        """
        Classify the sentiment of the given text.
        """
        pass
