from typing import override

import openai
from jinja2 import Template, Environment, meta
from itertools import product

from ._base import BaseFXClassifier, FXClassificationOutput

DEFAULT_PROMPT_TEMPLATE = """You are a financial news annotator. Determine the future directional sentiment for the specified target asset based on the provided articles.

# Labels
- UP: Asset expected to rise (e.g., for USDJPY, USD strengthens vs JPY → rate up)
- DOWN: Asset expected to fall (USD weakens vs JPY → rate down)
- FLAT: Range-bound / little change expected
- MIXED: Conflicting signals, no clear direction
- NO_SIGNAL: Asset mentioned but no directional cue

# Rules
1. Output ONLY a single valid JSON object. No extra text before/after.
2. For every label {UP, DOWN, FLAT, MIXED, NO_SIGNAL}, assign a probability in "scores". Each ∈ [0,1], and the sum must be 1 ± 0.01. If the sum deviates, renormalize before output.
3. "final_label" = the label with the highest probability.
4. FX convention: “rate UP” means the first currency in the pair appreciates vs the second (e.g., USDJPY up = USD↑ / JPY↓).
5. If the article mentions the asset but gives no directional hint, NO_SIGNAL should dominate.
6. "evidence_sentences": short verbatim snippets (≤ 30 words each, up to 3 sentences) directly supporting the classification.
7. Do NOT output internal reasoning or analysis; no chain-of-thought.
8. Use 3 decimal places for probabilities.

# Examples

## Shot 1
Input:
  [ASSET]: USDJPY
  [TEXT]: U.S. CPI came in hotter than expected, pushing Treasury yields higher. Traders now price out near-term Fed cuts, boosting the dollar against major peers.
Output:
{
  "asset": "USDJPY",
  "scores": {
    "UP": 0.86,
    "DOWN": 0.02,
    "FLAT": 0.05,
    "MIXED": 0.05,
    "NO_SIGNAL": 0.02
  },
  "final_label": "UP",
  "evidence_sentences": [
    "U.S. CPI came in hotter than expected",
    "boosting the dollar against major peers"
  ],
  "notes_for_humans": "Higher CPI → fewer Fed cuts → stronger USD"
}


## Shot 2
Input:
  [ASSET]: USDJPY
  [TEXT]: Speculation grew that the Bank of Japan could raise rates sooner than markets expect, prompting yen buying in afternoon trading.
Output:
{
  "asset": "USDJPY",
  "scores": {
    "UP": 0.05,
    "DOWN": 0.80,
    "FLAT": 0.05,
    "MIXED": 0.06,
    "NO_SIGNAL": 0.04
  },
  "final_label": "DOWN",
  "evidence_sentences": [
    "Speculation grew that the Bank of Japan could raise rates sooner",
    "prompting yen buying"
  ],
  "notes_for_humans": "BoJ hike talk → yen buying → USDJPY down"
}


## Shot 3
Input:
  [ASSET]: USDJPY
  [TEXT]: Analysts expect the pair to remain range-bound this week, with no major data releases on the calendar.
Output:
{
  "asset": "USDJPY",
  "scores": {
    "UP": 0.08,
    "DOWN": 0.07,
    "FLAT": 0.72,
    "MIXED": 0.05,
    "NO_SIGNAL": 0.08
  },
  "final_label": "FLAT",
  "evidence_sentences": [
    "expect the pair to remain range-bound this week"
  ],
  "notes_for_humans": "No catalysts; sideways view"
}


## Shot 4
Input:
  [ASSET]: USDJPY
  [TEXT]: Stronger U.S. jobs data lifted the dollar early, but dovish comments from Fed officials later capped gains, leaving traders divided.
Output:
{
  "asset": "USDJPY",
  "scores": {
    "UP": 0.18,
    "DOWN": 0.14,
    "FLAT": 0.05,
    "MIXED": 0.60,
    "NO_SIGNAL": 0.03
  },
  "final_label": "MIXED",
  "evidence_sentences": [
    "Stronger U.S. jobs data lifted the dollar early",
    "dovish comments from Fed officials later capped gains"
  ],
  "notes_for_humans": "Bullish and bearish drivers coexist"
}

## Shot 5
Input:
  [ASSET]: USDJPY
  [TEXT]: Tesla reported record quarterly deliveries, beating analysts' expectations.
Output:
{
  "asset": "USDJPY",
  "scores": {
    "UP": 0.01,
    "DOWN": 0.01,
    "FLAT": 0.02,
    "MIXED": 0.01,
    "NO_SIGNAL": 0.95
  },
  "final_label": "NO_SIGNAL",
  "evidence_sentences": [],
  "notes_for_humans": "No FX direction mentioned"
}

# Tasks

You will be given a list of pairs of articles and assets. For each pair, classify the sentiment of the article with respect to the asset. The output should be a JSON object with the following structure:
Input:
{{pairs}}

Output:
"""

class LLMFXClassifier(BaseFXClassifier):
    input_variables = ["pairs"]
    default_prompt_template = DEFAULT_PROMPT_TEMPLATE

    def __init__(self, *, prompt_template: str = None):
        self.prompt_template = prompt_template or self.default_prompt_template
        self._check_prompt()

    def _check_prompt(self):
        """
        Check if the prompt template is valid.
        """
        env = Environment()
        ast = env.parse(self.prompt_template)
        referenced = meta.find_undeclared_variables(ast)

        for keyword in self.input_variables:
            if keyword not in referenced:
                raise ValueError(
                    f"Keyword '{keyword}' not found in Jinja2 prompt template."
                )

    def create_prompt(self, *, assets: list[str], texts: list[str], asset_text_pairs: list[tuple[str, str]] = None) -> str:
        """
        Create a prompt for the LLM.
        """
        if not asset_text_pairs:
          if not assets or not texts:
            raise ValueError("Either 'asset_text_pairs' or both 'assets' and 'texts' must be provided.")
          asset_text_pairs = list(product(assets, texts))

        pairs_str = "\n".join(f"{asset}: {text}" for asset, text in asset_text_pairs)
        template = Template(self.prompt_template)
        return template.render(pairs=pairs_str)

class OpenAIFXClassifier(LLMFXClassifier):
    def __init__(self, *, client: openai.Client, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = client

    @override
    def classify(
        self, *, assets: list[str], texts: list[str], asset_text_pairs: list[tuple[str, str]] = None, model: str = "gpt-4o"
    ) -> FXClassificationOutput:
        """
        Classify the sentiment of the given text using OpenAI's API.
        """
        prompt = self.create_prompt(assets=assets, texts=texts, asset_text_pairs=asset_text_pairs)
        response = self.client.responses.parse(
            model=model,
            input=prompt,
            text_format=FXClassificationOutput,
        )
        return response.output_parsed
