import sys

import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

sys.path.append("..")  # Adjust the path to the src directory

from src.fx import OpenAIFXClassifier

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2025-03-01-preview",
)

assets = ["USD/JPY"]  # Example assets for FX classification

# load text data
df = pd.read_csv("../data/interim/sample.csv")
texts = []
for i in range(len(df)):
    bpdy = df.iloc[i]["body"]
    headline = df.iloc[i]["headline"]
    text = f"{headline}\n\n{bpdy}"
    texts.append(text)

print(f"Number of texts: {len(texts)}")

batch_size = 10  # Number of texts to classify in each batch

# Initialize the FX classifier
fx_classifier = OpenAIFXClassifier(client=client)
# Classify the FX texts
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    print(f"Classifying batch {i // batch_size + 1} with {len(batch_texts)} texts...")

    # Classify the texts
    output = fx_classifier.classify(assets=assets, texts=batch_texts)

    print(len(output.results), "results returned.")

    # Print the results
    for result in output.results:
        print(f"Asset: {result.asset}")
        print(f"Article: {result.article}")
        print(f"Final Label: {result.final_label}")
        print(f"Scores: {result.scores}")
        print(f"Evidence Sentences: {result.evidence_sentences}")
        print(f"Notes for Humans: {result.notes_for_humans}\n")

    break
