import sys

from openai import AzureOpenAI
from dotenv import load_dotenv

sys.path.append("..")  # Adjust the path to the src directory

from src.sentiment import OpenAISentimentClassifier

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2025-03-01-preview",
)

# Create a sentiment classifier
classifier = OpenAISentimentClassifier(
    client=client,
)


# Classify a sample text
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."  # noqa: E501
model = "gpt-4o"  # Specify the model to use for classification
output = classifier.classify(
    text,
    model=model,
)

for prediction in output.predictions:
    print(f"Label: {prediction.label}, Confidence: {prediction.confidence:.2f}")
