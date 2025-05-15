from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, create_model


def make_output_schema(topics: list[str]):
    """
    Create a Pydantic model for the output schema depending on the topics.
    """
    _Topic = Literal[tuple(topics)]

    PredictedTopic = create_model(
        "PredictedTopic",
        __base__=BaseModel,
        topic=(_Topic, Field(..., description="Predicted topic")),
        confidence=(float, Field(..., description="Confidence score")),
    )

    return create_model(
        "PredictedTopics",
        __base__=BaseModel,
        topics=(
            list[PredictedTopic],
            Field(..., description="List of predicted topics"),
        ),
    )


class BaseTopicClassifier(ABC):
    """
    Base class for topic classifiers.
    """

    @abstractmethod
    def classify(self, text: str):
        """
        Classify the given text into a topic.
        """
        pass

    @abstractmethod
    def topic_list(self) -> list[str]:
        """
        List of topics that the classifier can classify.
        """
        pass
