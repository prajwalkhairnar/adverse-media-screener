
# Adverse Media Screener: An LLM-Powered Compliance Engine

The **Adverse Media Screener** is a robust, AI-powered system designed for financial institutions to automate the Know Your Customer (KYC) and Anti-Money Laundering (AML) compliance screening process. It uses a **LangGraph** state machine to orchestrate multiple Large Language Model (LLM) calls, ensuring high accuracy, traceability, and auditability in classifying news articles related to screened individuals.

This system adheres to strict compliance requirements by utilizing **structured Pydantic models** for all inputs, outputs, and intermediate states.

---

## Key Features

* **LangGraph Orchestration:** A defined, multi-step workflow handles fetching, extracting, matching, sentiment analysis, and reporting, providing clear logic flow and easy maintenance.
* **Compliance-First Output:** All LLM outputs are validated against **Pydantic schemas** (`src/models/outputs.py`) for reliable, machine-readable, and auditable results (Section 3.2).
* **LLM Agnostic:** Supports multiple providers including **Groq, OpenAI, and Anthropic**, complete with a **fallback mechanism** and dedicated **Cost Tracker** (`src/llm/`).
* **High Traceability (Audit Trail):** Every LLM call is logged and audited for **token usage, cost, and latency** (Section 8.2), with full support for **LangSmith** tracing (`src/observability/tracer.py`).
* **Deterministic Rules:** Incorporates non-LLM, rule-based checks, such as **Age Alignment Verification** (Section 6.2), to increase the reliability of the Name Matching step.

---

## Architecture Overview

The system is built around a **LangGraph** state machine, which is governed by a central `ScreeningState` Pydantic model. The workflow is split into five distinct nodes:



| Component | Responsibility | Key Files |
| :--- | :--- | :--- |
| **State** | The central data bus for the entire workflow. A single dictionary object that is passed and mutated by nodes. | `src/graph/state.py` |
| **Nodes** | The primary orchestration logic. Nodes receive the state, run deterministic checks, invoke the corresponding LLM Chain, handle errors, track costs, and update the state. | `src/nodes/*.py` |
| **Chains** | The pure LLM logic (Prompt + LLM + Structured Parser). These are reusable components decoupled from the graph state management. | `src/chains/*.py` |
| **Factory** | Handles configuration and initialization of LLM clients (OpenAI, Groq, Anthropic) with required compliance settings (e.g., `temperature=0`). | `src/llm/factory.py` |
| **Cost Tracker** | Audits token usage and calculates the estimated cost for every LLM step, providing data for the final audit report. | `src/llm/cost_tracker.py` |

---

## Installation and Setup

### Prerequisites

* Python 3.11 or later
* Access to at least one supported LLM API (Groq, OpenAI, or Anthropic).

### 1. Clone the repository

```bash
git clone https://github.com/prajwalkhairnar/adverse-media-screener.git
cd adverse-media-screener
````

### 2\. Set up the Python environment

We recommend using a virtual environment.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies (from pyproject.toml / requirements.txt)
pip install -r requirements.txt
```

### 3\. Configure Environment Variables

Create a file named **`.env`** in the project root based on the provided `.env.example`.

```dotenv
# .env

# =============================================================================
# Core LLM Configuration (Choose at least one)
# =============================================================================
DEFAULT_LLM_PROVIDER=groq # groq, openai, or anthropic
LLM_TEMPERATURE=0.0       # Set to 0.0 for deterministic, compliant behavior

GROQ_API_KEY="YOUR_GROQ_API_KEY"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# Model names (Optional overrides)
GROQ_MODEL="llama-3.3-70b-versatile"
OPENAI_MODEL="gpt-4o-mini"
ANTHROPIC_MODEL="claude-sonnet-4"

# =============================================================================
# Observability (LangSmith)
# =============================================================================
# Set these to enable tracing for every LLM call (Highly Recommended for Audit)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
LANGCHAIN_PROJECT="adverse-media-screening-v1"

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=info # debug, info, warning, error
```

-----

## Usage

The primary entry point is `src/main.py`, which is designed to be run via a simple CLI (using the `click` library).

### Running a Screening Query

To screen a person against a specific article:

```bash
python src/main.py screen \
    --name "John Smith" \
    --dob "1985-07-25" \
    --url "[https://mock-url.com/news/john-smith-investigation](https://mock-url.com/news/john-smith-investigation)" \
    --provider "openai"  # Optional: overrides DEFAULT_LLM_PROVIDER
```

**Example Output (Console):**

The script will log the step-by-step execution path:

```
[INFO] [ArticleFetcher] Attempting to fetch and parse article: ...
[INFO] [ExtractionNode] Running Entity Extraction Node...
[INFO] [ExtractionNode] Successfully extracted 1 person entities.
[INFO] [MatchingNode] Running Name Matching Node...
[INFO] [MatchingNode] Match decision: MATCH (Confidence: HIGH, Prob: 0.95)
[INFO] [SentimentNode] Running Sentiment Analysis Node...
[INFO] [SentimentNode] Sentiment Result: NEGATIVE, Adverse: True, Severity: HIGH
[INFO] [ReportNode] Running Report Generation Node...
[INFO] [ReportNode] Report generation complete. Finalizing state.
```

The final structured report will be printed to the console (and optionally saved).

-----

## Testing

The test suite ensures that all deterministic logic and LLM orchestration flow correctly by **mocking all external dependencies** (network calls and LLM responses).

Run tests from the project root:

```bash
pytest
```

This will execute tests for the `ArticleFetcher`, `EntityExtractionNode`, and a full `test_integration.py` workflow run using the mock data in `tests/test_cases.json`.

-----

## Compliance and Audit

All compliance requirements are met via structured outputs:

  * **Final Report Schema:** The final output is an instance of the `ScreeningResult` Pydantic model, guaranteeing all required fields are present (Section 3.3).
  * **Audit Trail:** The `ProcessingMetadata` and detailed `llm_calls` logs ensure a complete record of every step and resource consumed.
  * **Report Template:** The structure of the final human-readable report is defined in the LLM prompt (`config/prompts.py`) to ensure all compliance points (decision, evidence, reasoning) are covered.

For a detailed breakdown of the expected final report structure, see: [REPORT.md](https://www.google.com/search?q=docs/REPORT.md).


