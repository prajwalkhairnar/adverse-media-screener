# Adverse Media Screener: Agent Guide

This document provides a technical overview of the Adverse Media Screener system, designed to help AI agents understand the architecture, setup, and testing workflows.

## 1. System Overview

The **Adverse Media Screener** is a compliance engine that automates KYC/AML screening using a **LangGraph** state machine. It orchestrates LLM calls to classify news articles related to screened individuals.

### Core Architecture
- **State Machine**: Managed by `src/graph/state.py`. The workflow passes a `ScreeningState` object through multiple nodes.
- **Nodes** (`src/nodes/`):
    - `ArticleFetcher`: Retrieves article content.
    - `ExtractionNode`: Extracts entities (Person, Organization).
    - `MatchingNode`: Matches extracted entities against the screened individual (uses fuzzy matching & LLM).
    - `SentimentNode`: Analyzes sentiment and adverse media risk.
    - `ReportNode`: Generates the final compliance report.
- **LLM Abstraction**: `src/llm/factory.py` handles provider switching (Groq, OpenAI, Anthropic) and enforces compliance settings (e.g., `temperature=0`).

## 2. Setup Instructions

### Prerequisites
- Python 3.11+
- API Keys for at least one LLM provider (Groq, OpenAI, or Anthropic).

### Installation
1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd adverse-media-screener
    ```
2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration**:
    - Copy `.env.example` to `.env`.
    - Set `DEFAULT_LLM_PROVIDER` (e.g., `groq`, `openai`).
    - Add corresponding API keys (e.g., `GROQ_API_KEY`).

## 3. Testing Guide

### Single Screening (Manual Test)
To test a single case, use the CLI entry point `src/main.py`.

**Command:**
```bash
python -m src.main screen \
    --name "John Smith" \
    --dob "1985-07-25" \
    --url "https://example.com/article" \
    --provider "groq"
```

**Output:**
- Console logs showing the graph execution path.
- Final structured report printed to stdout.

### Bulk / End-to-End Testing (Automated)
The `run_e2e.py` script runs a suite of real-world scenarios to validate the entire pipeline.

**Command:**
```bash
python run_e2e.py
```

**How it works:**
1.  **Test Cases**: Defined in `TEST_CASES` list within `run_e2e.py`. Each case includes:
    - `name`, `dob`, `url`
    - `expected_decision` (MATCH, NO_MATCH, UNCERTAIN)
2.  **Execution**: Iterates through cases, calling `src/main.py` via subprocess.
3.  **Validation**: Parses the output to compare the actual decision against the expected decision.
4.  **Metrics**: Calculates Recall, Precision, and False Negative Rate at the end.

**Adding a New Test Case:**
Modify `run_e2e.py` and append to the `TEST_CASES` list:
```python
{
    "name": "New Target",
    "dob": "1990-01-01",
    "url": "https://news-site.com/article",
    "description": "Testing specific adverse scenario...",
    "expected_decision": "MATCH"
}
```

## 4. Key Files Reference

| File | Purpose |
| :--- | :--- |
| `src/main.py` | CLI entry point for single screens. |
| `run_e2e.py` | Script for bulk/E2E testing and metrics calculation. |
| `src/graph/graph.py` | Defines the LangGraph workflow structure. |
| `src/models/` | Pydantic models for inputs, outputs, and state. |
| `README.md` | User-facing documentation. |
