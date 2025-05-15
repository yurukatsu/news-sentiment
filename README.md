# News Sentiment Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?logo=OpenAI&logoColor=white)](https://openai.com/)

ニューステキストの感情分析とトピック分類を行うためのPythonライブラリです。金融ニュースの分析に特化しており、複数の手法を提供しています。

## 特徴

- **感情分析**: ニューステキストをポジティブ、ネガティブ、ニュートラルに分類
  - FinBERT（Hugging Face）を使用した高速な分析
  - OpenAI GPT-4o などの大規模言語モデル（LLM）を使用した高精度な分析
- **トピック分類**: ニュースを指定したトピックカテゴリに分類
  - カスタマイズ可能なトピックリスト
  - OpenAI LLM を使用した柔軟な分類

## インストール

### 前提条件

- Python 3.11以上

### インストール手順

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/news-sentiment.git
cd news-sentiment

# 依存関係のインストール
pip install -e .
```

または、直接インストールする場合:

```bash
pip install git+https://github.com/yourusername/news-sentiment.git
```

## 環境設定

Azure OpenAI を使用する場合は、`.env` ファイルを作成し、以下の環境変数を設定してください:

```
AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
OPENAI_API_VERSION="2025-03-01-preview"
```

## 使用例

### FinBERT を使用した感情分析

```python
from src.sentiment import FinBERTClassifier

# 感情分析器の作成
classifier = FinBERTClassifier()

# テキストの分析
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."
output = classifier.classify(text)

# 結果の表示
for prediction in output.predictions:
    print(f"Label: {prediction.label}, Confidence: {prediction.confidence:.2f}")
```

### OpenAI LLM を使用した感情分析

```python
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.sentiment import OpenAISentimentClassifier

# 環境変数の読み込み
load_dotenv(override=True)

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(
    api_version="2025-03-01-preview",
)

# 感情分析器の作成
classifier = OpenAISentimentClassifier(
    client=client,
)

# テキストの分析
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."
model = "gpt-4o"  # 使用するモデルを指定
output = classifier.classify(
    text,
    model=model,
)

# 結果の表示
for prediction in output.predictions:
    print(f"Label: {prediction.label}, Confidence: {prediction.confidence:.2f}")
```

### トピック分類

```python
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.topic import OpenAITopicClassifier

# 環境変数の読み込み
load_dotenv(override=True)

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(
    api_version="2025-03-01-preview",
)

# トピック分類器の作成
topics = [
    "sports",
    "politics",
    "technology",
    "economy",
    "finance",
]  # 分類するトピックを定義
partial_variables = {
    "topics": topics,
    "n": 3,  # 返すトピックの数
}
classifier = OpenAITopicClassifier(
    topics=topics,
    partial_variables=partial_variables,
    client=client,
)

# テキストの分析
text = "The stock market is experiencing a significant downturn, with major indices falling sharply."
model = "gpt-4o"  # 使用するモデルを指定
predictions = classifier.classify(
    text,
    model=model,
)

# 結果の表示
for pred in predictions.topics:
    print(f"Topic: {pred.topic}, Score: {pred.confidence}")
```

## アーキテクチャ

このライブラリは以下のコンポーネントで構成されています：

- `src/sentiment/`: 感情分析モジュール
  - `_base.py`: 基本クラスと型定義
  - `hugging_face.py`: Hugging Face モデルを使用した実装
  - `llm.py`: LLM を使用した実装
- `src/topic/`: トピック分類モジュール
  - `_base.py`: 基本クラスと型定義
  - `llm.py`: LLM を使用した実装

## 貢献

バグ報告や機能リクエストは GitHub Issues にお願いします。プルリクエストも歓迎します。

## ライセンス

[MIT](https://opensource.org/licenses/MIT)
