from string import Formatter
from typing import override

import openai
from ._base import BaseSentimentClassifier, SentimentClassifierOutput


class LLMSentimentClassifier(BaseSentimentClassifier):
    input_variables = ["text"]
    default_prompt_template = """You are a news sentiment classifier.
Given a news headline or short excerpt, return a JSON array containing exactly these three labels—"positive", "neutral", and "negative"—each with a confidence score (a float between 0 and 1). Sort the entries by descending confidence.

-- Format --
Input: "<news text>"
Output:
    -   label: "positive"
        confidence: <float>
    -   label: "neutral"
        confidence: <float>
    -   label: "negative"
        confidence: <float>

-- Examples --
Input: "Stocks surge as better-than-expected jobs data boosts investor confidence"
Output:
    -   label: "positive"
        confidence: 0.92
    -   label: "neutral"
        confidence: 0.05
    -   label: "negative"
        confidence: 0.03

Input: "Federal Reserve leaves interest rates unchanged, citing mixed economic signals"
Output:
    -   label: "neutral"
        confidence: 0.85
    -   label: "positive"
        confidence: 0.10
    -   label: "negative"
        confidence: 0.05

Input: "Company X recalls its flagship product after reports of malfunction and safety concerns"
Output:
    -   label: "negative"
        confidence: 0.90
    -   label: "neutral"
        confidence: 0.07
    -   label: "positive"
        confidence: 0.03

-- Now classify the following --
Input: "{text}"
Output:
"""  # noqa: E501

    def __init__(self, *, prompt_template: str = None):
        self.prompt_template = prompt_template or self.default_prompt_template
        self._check_prompt()

    def _check_prompt(self):
        """
        Check if the prompt template is valid.
        """
        formatter = Formatter()
        placeholders = [name for _, name, _, _ in formatter.parse(self.prompt_template)]
        for keyword in self.input_variables:
            if keyword not in placeholders:
                raise ValueError(
                    f"Keyword '{keyword}' not found in prompt template placeholders."
                )

    def create_prompt(self, text: str) -> str:
        """
        Create a prompt for the LLM.
        """
        return self.prompt_template.format(text=text)


class OpenAISentimentClassifier(LLMSentimentClassifier):
    def __init__(self, *, client: openai.Client, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = client

    @override
    def classify(
        self, text: str, *, model: str = "gpt-4o"
    ) -> SentimentClassifierOutput:
        """
        Classify the sentiment of the given text using OpenAI's API.
        """
        prompt = self.create_prompt(text)
        response = self.client.responses.parse(
            model=model,
            input=prompt,
            text_format=SentimentClassifierOutput,
        )
        return response.output_parsed
