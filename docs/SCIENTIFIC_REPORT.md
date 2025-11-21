# Adverse Media Screener: Methodology & Future Roadmap

## 1. Scientific Approach & Methodology

The core objective of this system is to maximize **Recall** (minimizing false negatives) while maintaining high **Precision** (minimizing false positives) to reduce analyst workload. In a compliance context, missing a true adverse media match (False Negative) is a critical failure, whereas flagging a non-match for review (False Positive) is a manageable operational cost.

### Architecture: Deterministic Orchestration with Probabilistic Intelligence
We utilized **LangGraph** to build a state machine that decouples the workflow into distinct, testable nodes. This allows for a "scientific" pipeline where each step filters the data before passing it to the next, rather than a single "black box" LLM call.

1.  **Entity Extraction (Recall Focus):**
    *   *Goal:* Extract *every* person mentioned in the article to ensure no potential match is missed.
    *   *Technique:* The prompt explicitly instructs the model to be over-inclusive.

2.  **Name Matching (Precision Focus):**
    *   *Goal:* Compare the query individual against extracted entities using both rule-based and semantic checks.
    *   *Technique:*
        *   **Rule-Based Filter:** A deterministic `verify_age_alignment` function checks DOBs against ages mentioned in the text (e.g., "45 years old") to instantly discard impossible matches.
        *   **Chain-of-Thought Prompting:** The LLM is forced to output `<reasoning_steps>` before making a decision, reducing hallucination and ensuring it considers nicknames (e.g., "Jim" vs "James") and cultural naming conventions.

3.  **Sentiment Analysis (Contextual Understanding):**
    *   *Goal:* Determine if the match is actually "adverse".
    *   *Technique:* We distinguish between "negative news" (e.g., a business failure) and "adverse media" (e.g., fraud, money laundering) by categorizing findings into specific risk buckets (Legal, Financial, Ethical).

## 2. Test Strategy & Results

### Model Configuration
*   **Provider:** Groq
*   **Model:** `moonshotai/kimi-k2-instruct-0905`
*   **Note:** This model was chosen for its strong instruction-following capabilities and large context window. Results may vary slightly with other models (e.g., GPT-4o, Claude 3.5 Sonnet), particularly in how they interpret ambiguous "Uncertain" cases.

### Test Cases
We designed a suite of End-to-End (E2E) test cases to validate the system against common failure modes:

1.  **Elizabeth Holmes:** High-profile adverse match (Testing basic functionality).
2.  **Satya Nadella:** Neutral/Positive match (Testing sentiment differentiation).
3.  **Chris Smith:** Common name with ambiguous details (Testing "Uncertainty" handling).
4.  **Paul Anderson:** Adverse article but client not mentioned (Testing entity resolution).

### Current Metrics
Based on the `run_e2e.py` test suite execution:

*   **Recall:** **1.0 (100%)** - The system successfully identified all true adverse cases.
*   **Precision:** **1.0 (100%)** - The system did not flag any clear non-matches as adverse.

*Note: "Uncertain" results are treated as "Review Required" (Positive) for Recall calculations to ensure safety.*

---

## 3. Part 2: Automated Enrichment Plan

In many real-world scenarios, an article may mention a "John Smith" involved in fraud but lack a Date of Birth or middle name to confirm if it matches our client "John David Smith, DOB 1980-01-01".

To resolve these ambiguities automatically, we propose adding an **Enrichment Node** to the graph.

### Proposed Workflow

If the `MatchingNode` returns **UNCERTAIN** due to missing information (e.g., "Possible match, but age unknown"), the system will trigger the Enrichment workflow:

#### Step 1: Targeted Web Search (Search Tool)
Use a search API (e.g., SerpApi, Tavily) to find corroborating details linking the person in the article to the client's identifiers.

*   **Query Generation:** The LLM generates specific queries based on the missing info.
    *   *Example:* `"John Smith" fraud "New York" date of birth`
    *   *Example:* `"John Smith" "Acme Corp" middle name`

#### Step 2: Cross-Reference Professional Registries
If the article mentions an occupation (e.g., "Director at Acme Corp"), query structured databases:
*   **LinkedIn API / Proxy:** To confirm employment history and education.
*   **Corporate Registries (e.g., Companies House, OpenCorporates):** To verify directorships and retrieve listed DOBs (often available as Month/Year).

#### Step 3: Synthesize & Re-Evaluate
1.  Feed the newly retrieved snippets back into the `MatchingNode`.
2.  Update the `ScreeningState` with the new evidence.
3.  Re-run the match assessment.

### Safety Mechanisms
*   **Source Credibility:** The Enrichment Node will prioritize "primary sources" (government registries, official company pages) over secondary sources (blogs, forums).
*   **Provenance Tracking:** All external data fetched will be logged with its source URL in the final report to allow the analyst to verify the "enriched" facts.

---

## 4. Future Optimizations

### Performance & Scalability
*   **Sequential vs. Parallel Execution:** Currently, LLM calls (Extraction → Matching → Sentiment) are executed sequentially for simplicity and clear state management (MVP architecture).
*   **Bulk Processing:** For high-volume production environments, the **Name Matching** stage can be optimized by processing multiple entities in parallel (batching) or using map-reduce patterns in LangGraph. This would significantly reduce end-to-end latency when an article contains many extracted entities.
