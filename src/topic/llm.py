from string import Formatter
from typing import Any

import openai
from typing import override

from ._base import BaseTopicClassifier, make_output_schema


class PromptTemplate:
    """
    A class to represent a prompt template for LLM.
    """

    def __init__(self, template: str, partial_variables: dict[str, Any] = None):
        self.template = template
        self.partial_variables = partial_variables or {}

    def add_variable(self, name: str, value: Any):
        """
        Add a variable to the template.
        """
        self.partial_variables[name] = value

    def has_placeholders(self, keywords: list[str]):
        """
        Check if the template has placeholders for the given keywords.
        """
        formatter = Formatter()
        placeholders = [name for _, name, _, _ in formatter.parse(self.template)]

        for keyword in keywords:
            if keyword not in placeholders:
                raise ValueError(
                    f"Keyword '{keyword}' not found in template placeholders."
                )

    def format(self, **input_variables):
        """
        Format the template with the given keyword arguments.
        """
        variables = {**self.partial_variables, **input_variables}
        return self.template.format(**variables)


class LLMTopicClassifier(BaseTopicClassifier):
    default_prompt_template: str = """You are a news topic classifier specialized for foreign-exchange prediction.
Given:
    - A list of candidate topics
    - A news headline or short excerpt
    - A number N
Return a list of the top N topics (from the provided list) with their confidence scores (floats between 0 and 1), sorted descending by confidence.

--Format--
Input:
    Topics: ["<topic1>", "<topic2>", "<topic3>", ...]
    Text: "<news text>"
    N: <integer>

Output:
    -   "topic": "<topic1>"
        "confidence": <float1>
    -   "topic": "<topic2>"
        "confidence": <float2>
    -   "topic": "<topic3>"
        "confidence": <float3>
    … up to N entries …

-- Examples --
Here are some examples under the following topics:

["inflation (CPI, PPI)", "frb", "Labor", "other"]

Input:
    Topics: ["other", "frb", "inflation (CPI, PPI)", "Labor"]
    Text: "The Federal Reserve announces it will hold interest rates steady at 0.25%"
    N: 2
Output:
    -   "topic": "frb"
        "confidence": 0.95
    -   "topic": "inflation (CPI, PPI)"
        "confidence": 0.05

Input:
    Topics: ["other", "frb", "inflation (CPI, PPI)", "Labor"]
    Text: "April’s Consumer Price Index (CPI) rose 3.4% year-over-year"
    N: 1
Output:
    -   "topic": "inflation (CPI, PPI)"
        "confidence": 0.99

Input:
    Topics: ["other", "frb", "inflation (CPI, PPI)", "Labor"]
    Text: "Apple shares jump as the new iPhone sales exceed forecasts"
    N: 3
Output:
    -   "topic": "other"
        "confidence": 0.85
    -   "topic": "frb"
        "confidence": 0.10
    -   "topic": "Labor"
        "confidence": 0.05

-- Now classify the following --
Input:
    Topics: {topics}
    Text: "{text}"
    N: {n}
Output:"""  # noqa: E501

    def __init__(
        self,
        topics: list[str],
        partial_variables: dict[str, Any],
        *,
        prompt_template: str = None,
        text_placeholder: str = "text",
        partial_placeholders: list[str] = ["topics", "n"],
    ):
        """
        Initialize the LLMTopicClassifier.

        Args:
            topics (list[str]): List of topics to classify.
            partial_variables (Dict[str, Any]): Dictionary of partial variables for the prompt template.
            prompt_template (str): The prompt template to use.
            text_placeholder (str): The placeholder for the text in the prompt template.
            partial_placeholders (list[str]): List of placeholders in the prompt template.
        """  # noqa: E501
        self.topics = topics
        self.prompt_template = PromptTemplate(
            prompt_template or self.default_prompt_template
        )
        self.prompt_template.has_placeholders([text_placeholder] + partial_placeholders)
        for placeholder in partial_placeholders:
            if placeholder in partial_variables:
                self.prompt_template.add_variable(
                    placeholder, partial_variables[placeholder]
                )
            else:
                raise ValueError(
                    f"Keyword '{placeholder}' not found in partial variables.",
                    "The keyword must be in the prompt template.",
                )
        self.text_placeholder = text_placeholder
        self.output_schema = make_output_schema(topics)

    def create_prompt(self, text: str) -> str:
        """
        Create a prompt for the LLM.
        """
        input_variables = {
            self.text_placeholder: text,
        }
        return self.prompt_template.format(**input_variables)

    @override
    def classify(self, text: str):
        """
        Classify the given text into a topic using LLM.
        """
        pass

    @property
    @override
    def topic_list(self) -> list[str]:
        """
        List of topics that the classifier can classify.
        """
        return self.topics


class OpenAITopicClassifier(LLMTopicClassifier):
    def __init__(
        self,
        topics: list[str],
        partial_variables: dict[str, Any],
        *,
        client: openai.Client,
        prompt_template: str = None,
        text_placeholder: str = "text",
        partial_placeholders: list[str] = ["topics", "n"],
    ):
        """
        Initialize the OpenAITopicClassifier.

        Args:
            topics (list[str]): List of topics to classify.
            partial_variables (Dict[str, Any]):
                Dictionary of partial variables for the prompt template.
            client (openai.Client): OpenAI client to use LLM.
            prompt_template (str): The prompt template to use.
            text_placeholder (str): The placeholder for the text in the prompt template.
            partial_placeholders (list[str]):
                List of placeholders in the prompt template.
        """
        super().__init__(
            topics,
            partial_variables,
            prompt_template=prompt_template,
            text_placeholder=text_placeholder,
            partial_placeholders=partial_placeholders,
        )
        self.client = client

    @override
    def classify(self, text, *, model: str = "gpt-4o-mini"):
        """
        Classify the given text into a topic using OpenAI LLM.
        """
        prompt = self.create_prompt(text)
        response = self.client.responses.parse(
            model=model,
            input=prompt,
            text_format=self.output_schema,
        )
        return response.output_parsed
