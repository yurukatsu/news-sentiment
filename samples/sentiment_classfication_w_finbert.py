import sys

sys.path.append("..")  # Adjust the path to the src directory

from src.sentiment import FinBERTClassifier

# Create a sentiment classifier
classifier = FinBERTClassifier()


# Classify a sample text
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."  # noqa: E501
output = classifier.classify(text)

for prediction in output.predictions:
    print(f"Label: {prediction.label}, Confidence: {prediction.confidence:.2f}")
