"""
Prompt Engineering: Templates, CoT, Tool Calling, A/B Testing
Demonstrates systematic prompt optimization with measurable results.
"""

import json
import re
from dataclasses import dataclass
from string import Template
from typing import Any, Callable


# ─── Prompt Templates ─────────────────────────────────────────────────────────

TEMPLATES = {
    "zero_shot": Template(
        "Classify the sentiment of this text as positive, negative, or neutral.\n\nText: $text\n\nSentiment:"
    ),
    "few_shot": Template(
        """Classify sentiment as positive, negative, or neutral.

Examples:
Text: "The product exceeded my expectations!" → positive
Text: "Terrible experience, total waste of money." → negative
Text: "Item arrived on time." → neutral

Text: "$text" →"""
    ),
    "chain_of_thought": Template(
        """Analyze the sentiment of the following text step by step.

Text: "$text"

Steps:
1. Identify emotional words or phrases
2. Determine overall tone (positive/negative/neutral signals)
3. Consider context and nuance
4. Conclude with final sentiment

Analysis:"""
    ),
    "structured_output": Template(
        """Analyze this text and return a JSON object with fields:
- sentiment: "positive" | "negative" | "neutral"
- confidence: float 0-1
- key_phrases: list of 1-3 key phrases driving the sentiment
- reasoning: one-sentence explanation

Text: "$text"

JSON response:"""
    ),
    "self_correction": Template(
        """Classify the sentiment of this text.

Text: "$text"

Step 1 - Initial classification:
[classify]

Step 2 - Check your reasoning:
- Did you consider the full context?
- Are there any mixed signals?

Step 3 - Final answer (confirm or revise): """
    ),
}


# ─── Tool Definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_date",
        "description": "Get the current date and time",
        "parameters": {"type": "object", "properties": {}},
    },
]


# ─── Tool Executor ────────────────────────────────────────────────────────────

def execute_tool(name: str, params: dict) -> str:
    """Execute tool calls from LLM. In production, these call real APIs."""
    import datetime

    if name == "search_knowledge_base":
        query = params.get("query", "")
        return json.dumps({
            "results": [
                {"content": f"Result 1 for '{query}': relevant info here", "score": 0.92},
                {"content": f"Result 2 for '{query}': additional context", "score": 0.85},
            ]
        })
    elif name == "calculate":
        expr = params.get("expression", "0")
        try:
            # Safe eval for demo (use ast.literal_eval or dedicated math lib in prod)
            allowed = set("0123456789+-*/()., ")
            if all(c in allowed for c in expr):
                result = eval(expr)  # noqa: S307
                return json.dumps({"result": result})
        except Exception:
            pass
        return json.dumps({"error": "Invalid expression"})
    elif name == "get_current_date":
        return json.dumps({"date": datetime.datetime.now().isoformat()})
    return json.dumps({"error": f"Unknown tool: {name}"})


# ─── A/B Test Framework ───────────────────────────────────────────────────────

@dataclass
class PromptVariant:
    name: str
    template: Template
    results: list[dict] = None

    def __post_init__(self):
        self.results = []

    def render(self, **kwargs) -> str:
        return self.template.substitute(**kwargs)

    def record(self, output: str, ground_truth: str, cost: float = 0.0, latency_ms: float = 0.0):
        correct = ground_truth.lower() in output.lower()
        self.results.append({
            "output": output,
            "correct": correct,
            "cost": cost,
            "latency_ms": latency_ms,
        })

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r["correct"] for r in self.results) / len(self.results)

    @property
    def avg_cost(self) -> float:
        if not self.results:
            return 0.0
        return sum(r["cost"] for r in self.results) / len(self.results)


def ab_test_prompts(test_cases: list[dict], fake_llm: Callable) -> None:
    """
    Test multiple prompt variants against labeled data.
    Reports: accuracy, cost, latency per variant.
    """
    variants = [
        PromptVariant("zero_shot", TEMPLATES["zero_shot"]),
        PromptVariant("few_shot", TEMPLATES["few_shot"]),
        PromptVariant("chain_of_thought", TEMPLATES["chain_of_thought"]),
        PromptVariant("structured_output", TEMPLATES["structured_output"]),
    ]

    for case in test_cases:
        for variant in variants:
            prompt = variant.render(text=case["text"])
            output = fake_llm(prompt, case["label"])
            variant.record(output, case["label"], cost=len(prompt) * 0.000005)

    print("\n=== Prompt A/B Test Results ===")
    print(f"{'Variant':<20} {'Accuracy':>10} {'Avg Cost':>12}")
    print("-" * 45)
    for v in sorted(variants, key=lambda x: -x.accuracy):
        print(f"{v.name:<20} {v.accuracy:>9.1%} ${v.avg_cost:>11.6f}")


# ─── Structured Output Parser ────────────────────────────────────────────────

def parse_structured_output(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": text}


# ─── Demo ────────────────────────────────────────────────────────────────────

def fake_llm(prompt: str, ground_truth: str) -> str:
    """Simulate LLM with slight noise — for demo without API keys."""
    import random
    labels = ["positive", "negative", "neutral"]
    if random.random() < 0.80:  # 80% accuracy simulation
        return f"The sentiment is {ground_truth}."
    else:
        other = [l for l in labels if l != ground_truth]
        return f"The sentiment is {random.choice(other)}."


TEST_CASES = [
    {"text": "This product is absolutely amazing! Best purchase ever.", "label": "positive"},
    {"text": "Worst customer service I have ever experienced.", "label": "negative"},
    {"text": "The package arrived on Thursday as expected.", "label": "neutral"},
    {"text": "I love how intuitive this tool is!", "label": "positive"},
    {"text": "Complete waste of money. Do not buy.", "label": "negative"},
    {"text": "Product looks exactly like the photos.", "label": "neutral"},
]

if __name__ == "__main__":
    # Show prompt templates
    sample = "The AI assistant helped me solve a complex problem quickly!"
    print("=== Prompt Variants ===")
    for name, tmpl in list(TEMPLATES.items())[:2]:
        print(f"\n[{name}]")
        print(tmpl.substitute(text=sample))

    # Tool definitions
    print("\n=== Tool Definitions ===")
    for tool in TOOLS:
        print(f"  {tool['name']}: {tool['description']}")

    # Execute tools
    print("\n=== Tool Execution ===")
    result = execute_tool("calculate", {"expression": "2 ** 10 + 42"})
    print(f"calculate: {result}")
    result = execute_tool("get_current_date", {})
    print(f"current_date: {result}")

    # A/B test
    ab_test_prompts(TEST_CASES, fake_llm)

    # Structured output
    print("\n=== Structured Output Parsing ===")
    raw = '```json\n{"sentiment": "positive", "confidence": 0.92, "key_phrases": ["amazing", "best purchase"], "reasoning": "Strong positive language throughout."}\n```'
    parsed = parse_structured_output(raw)
    print(parsed)
