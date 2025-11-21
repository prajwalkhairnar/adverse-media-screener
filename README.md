
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
    --provider "groq"  # Optional: overrides DEFAULT_LLM_PROVIDER
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


## Running Full End-to-End Test Suites (E2E)

While unit tests rely on mocks for speed, a dedicated E2E test suite validates the entire pipeline against live external services (Article Fetcher and LLM APIs), ensuring real-world accuracy and stability.

The project includes the `run_e2e.py` script in the root directory to automate the execution of multiple real-world adverse media scenarios. This script executes your main CLI program for a defined list of cases.

### 1. Execution
Run the script from your project root:
```
python run_e2e.py
```
Upon completion, the pipeline will log the step-by-step progress for each test case directly to the console. You will also find the full human-readable (.txt) and structured audit (.json) reports saved in your project root for each executed scenario.

### 2. Modifying or Adding Test Cases
The test cases are defined directly within the `run_e2e.py` script in a dictionary list named `TEST_CASES`. To add a new E2E scenario, simply append a new dictionary to this list:

```
# The central list of scenarios for E2E testing
TEST_CASES = [
    # Case 1: Adverse Match (High-Risk Fraud)
    {"name": "Elizabeth Holmes", "dob": "1984-02-03", "url": "https://www.cnbc.com/2022/11/18/elizabeth-holmes-sentenced-to-more-than-11-years-in-prison.html"},
    
    # Case 2: Neutral/Non-Adverse Match
    {"name": "Tim Cook", "dob": "1960-11-01", "url": "https://www.apple.com/leadership/tim-cook.html"},
    
    # Case 3: Adverse Media, but Client Not Found (Testing Match Failure)
    {"name": "Jane Doe", "dob": "1990-09-12", "url": "https://www.cnn.com/2023/11/02/us/sam-bankman-fried-trial-verdict/index.html"},
    
    # --- ADD NEW CASES HERE ---
    {"name": "New Client Name", "dob": "YYYY-MM-DD", "url": "https://new-live-article.com/adverse-event"},
]
```

The script uses the `subprocess.run()` function to execute the main CLI command for each item in this list, making it easy to add, remove, or modify scenarios[web:2].


-----

## Compliance and Audit

All compliance requirements are met via structured outputs:

  * **Final Report Schema:** The final output is an instance of the `ScreeningResult` Pydantic model, guaranteeing all required fields are present (Section 3.3).
  * **Audit Trail:** The `ProcessingMetadata` and detailed `llm_calls` logs ensure a complete record of every step and resource consumed.
  * **Report Template:** The structure of the final human-readable report is defined in the LLM prompt (`config/prompts.py`) to ensure all compliance points (decision, evidence, reasoning) are covered.
