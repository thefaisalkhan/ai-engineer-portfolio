"""
ReAct Agent: Reasoning + Acting
Implements the ReAct pattern: Thought → Action → Observation → repeat.
Demonstrates: tool calling, multi-step reasoning, error recovery.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ─── Tool System ──────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: dict = field(default_factory=dict)

    def execute(self, **kwargs) -> str:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)}"

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ─── Tool Implementations ─────────────────────────────────────────────────────

def tool_search(query: str, top_k: int = 3) -> str:
    """Simulated search over knowledge base."""
    kb = {
        "rag": "RAG combines retrieval with LLM generation. Steps: index → retrieve → generate.",
        "mlops": "MLOps includes monitoring, CI/CD, drift detection, and automated retraining.",
        "transformer": "Transformers use self-attention to process sequences in parallel.",
        "vector database": "Vector DBs store embeddings for similarity search using ANN algorithms.",
        "langchain": "LangChain provides chains, agents, memory, and tool integrations for LLMs.",
        "python": "Python is the primary language for ML/AI development. Key libs: NumPy, PyTorch.",
    }
    results = []
    for k, v in kb.items():
        if any(word.lower() in k.lower() or word.lower() in v.lower()
               for word in query.split()):
            results.append(v)
    return "\n".join(results[:top_k]) if results else "No relevant results found."


def tool_calculate(expression: str) -> str:
    """Safe math evaluation."""
    # Only allow safe characters
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\,\*\*\%\^]+$", expression):
        return "ERROR: Invalid characters in expression"
    try:
        result = eval(expression.replace("^", "**"))  # noqa: S307
        return str(round(result, 6))
    except Exception as e:
        return f"ERROR: {e}"


def tool_get_date(_: str = "") -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_summarize(text: str, max_words: int = 50) -> str:
    """Extract first N words as summary."""
    words = text.split()[:max_words]
    return " ".join(words) + ("..." if len(text.split()) > max_words else "")


AVAILABLE_TOOLS: list[Tool] = [
    Tool(
        name="search",
        description="Search the knowledge base for information about AI, ML, and engineering topics.",
        func=tool_search,
        parameters={"query": "str", "top_k": "int (optional, default 3)"},
    ),
    Tool(
        name="calculate",
        description="Evaluate mathematical expressions. Example: '2 ** 10 + 42' or '(100 * 0.15) / 3'.",
        func=tool_calculate,
        parameters={"expression": "str"},
    ),
    Tool(
        name="get_date",
        description="Get the current date and time.",
        func=tool_get_date,
        parameters={},
    ),
    Tool(
        name="summarize",
        description="Summarize a piece of text to fewer words.",
        func=tool_summarize,
        parameters={"text": "str", "max_words": "int (optional, default 50)"},
    ),
]


# ─── ReAct Agent ─────────────────────────────────────────────────────────────

@dataclass
class Step:
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent pattern.
    In production: the 'think' step calls an LLM. Here we use a rule-based fake.
    """

    def __init__(self, tools: list[Tool], max_steps: int = 6):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _tool_list(self) -> str:
        return "\n".join(
            f"  - {t.name}: {t.description}"
            for t in self.tools.values()
        )

    def _think(self, question: str, history: list[Step], step_num: int) -> Step:
        """
        In production: send question + history + tools to LLM, parse response.
        Here: rule-based logic that simulates LLM reasoning.
        """
        q_lower = question.lower()

        # Step 1: route to appropriate tool
        if step_num == 1:
            if any(w in q_lower for w in ["what is", "how does", "explain", "describe"]):
                topic = question.replace("What is ", "").replace("How does ", "").strip("?").strip()
                return Step(
                    thought=f"I need to search for information about '{topic}'.",
                    action="search",
                    action_input={"query": topic},
                )
            elif any(w in q_lower for w in ["calculate", "compute", "what is", "how much"]):
                # Extract math-like parts
                nums = re.findall(r"[\d\+\-\*\/\(\)\.\s]+", question)
                expr = nums[0].strip() if nums else "1 + 1"
                return Step(
                    thought=f"This looks like a calculation. I'll compute: {expr}",
                    action="calculate",
                    action_input={"expression": expr},
                )
            elif "date" in q_lower or "time" in q_lower or "today" in q_lower:
                return Step(
                    thought="The user wants current date/time information.",
                    action="get_date",
                    action_input={},
                )
            else:
                return Step(
                    thought=f"Let me search for relevant information about this question.",
                    action="search",
                    action_input={"query": question},
                )

        # Step 2: synthesize answer from observations
        observations = [s.observation for s in history if s.observation]
        if observations:
            context = " ".join(observations[:3])
            return Step(
                thought=f"I have gathered relevant information. I can now answer the question.",
                final_answer=f"Based on my research: {context[:300]}{'...' if len(context) > 300 else ''}",
            )

        return Step(
            thought="I was unable to find relevant information.",
            final_answer="I don't have enough information to answer this question confidently.",
        )

    def run(self, question: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        history: list[Step] = []

        for step_num in range(1, self.max_steps + 1):
            print(f"\n[Step {step_num}]")

            step = self._think(question, history, step_num)
            print(f"  Thought: {step.thought}")

            # Final answer
            if step.final_answer:
                print(f"  Final Answer: {step.final_answer}")
                history.append(step)
                break

            # Execute tool
            if step.action and step.action in self.tools:
                print(f"  Action: {step.action}({step.action_input})")
                step.observation = self.tools[step.action].execute(**(step.action_input or {}))
                print(f"  Observation: {step.observation[:150]}...")
            elif step.action:
                step.observation = f"ERROR: Tool '{step.action}' not found"
                print(f"  Observation: {step.observation}")

            history.append(step)

        # Extract final answer
        final = next((s.final_answer for s in reversed(history) if s.final_answer), "No answer generated.")
        observations = [s.observation for s in history if s.observation]

        return {
            "question": question,
            "answer": final,
            "steps": len(history),
            "tools_used": [s.action for s in history if s.action],
            "observations": observations,
        }


# ─── Demo ────────────────────────────────────────────────────────────────────

QUESTIONS = [
    "What is RAG and how does it work?",
    "What is today's date?",
    "Calculate 2 ** 16 + 100",
    "How does a transformer model process sequences?",
]

if __name__ == "__main__":
    agent = ReActAgent(tools=AVAILABLE_TOOLS, max_steps=4)

    print("=== Available Tools ===")
    for tool in AVAILABLE_TOOLS:
        print(f"  {tool.name}: {tool.description[:60]}...")

    results = []
    for question in QUESTIONS:
        result = agent.run(question)
        results.append(result)

    print(f"\n{'='*60}")
    print("=== Session Summary ===")
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:150]}...")
        print(f"   Steps: {r['steps']}, Tools: {r['tools_used']}")
