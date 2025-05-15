import sys

from openai import AzureOpenAI
from dotenv import load_dotenv

sys.path.append("..")  # Adjust the path to the src directory

from src.topic import OpenAITopicClassifier

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2025-03-01-preview",
)

# Create a topic classifier
topics = [
    "sports",
    "politics",
    "technology",
    "economy",
    "finance",
]  # Define the topics for classification
partial_variables = {
    "topics": topics,
    "n": 3,  # Number of topics to return
}  # Define the partial variables for the classifier
classifier = OpenAITopicClassifier(
    topics=topics,
    partial_variables=partial_variables,
    client=client,
)  # Initialize the classifier with the specified topics and partial variables


# Classify a sample text
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."  # noqa: E501
model = "gpt-4o"  # Specify the model to use for classification
predictions = classifier.classify(
    text,
    model=model,
)

for pred in predictions.topics:
    # Output is the predicted topics and their confidence scores
    print(f"Topic: {pred.topic}, Score: {pred.confidence}")
