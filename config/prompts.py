"""
LLM prompts for adverse media screening system.

All prompts use XML tags for structured input/output to improve LLM comprehension
and reliability, especially with Claude models.
"""

# =============================================================================
# Entity Extraction
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting information about people from news articles for financial compliance screening.

Your task is to identify ALL people mentioned in articles and extract their identifying details with high precision."""

ENTITY_EXTRACTION_USER_PROMPT = """<task>
Extract all people mentioned in the following news article. For each person, identify any available details that could help verify their identity.
</task>

<article>
<url>{article_url}</url>
<title>{article_title}</title>
<source>{article_source}</source>
<publish_date>{publish_date}</publish_date>
<language>{language}</language>
<content>
{article_content}
</content>
</article>

<instructions>
1. Extract EVERY person mentioned, even if only briefly
2. Capture the exact name as written in the article (including titles like "Dr." or "Sir")
3. Note any age information (exact age or approximate like "in his 40s")
4. Record occupation, job title, or role if mentioned
5. Note location (city, country, region) if mentioned
6. Include a context snippet (1-2 sentences) showing how this person is described
7. If the same person is mentioned multiple times with different details, combine all information
8. Include ANY other identifying information (education, company, family relations, etc.)
</instructions>

<output_format>
Return a JSON array of person objects. Each object should have:
{{
  "full_name": "exact name as in article",
  "age": numeric age if mentioned (or null),
  "approximate_age_range": "description like 'in his 40s'" (or null),
  "occupation": "job title or role" (or null),
  "location": "city, country, or region" (or null),
  "other_details": ["any other identifying info"],
  "context_snippet": "1-2 sentences showing how person is described",
  "confidence": "high/medium/low" based on clarity of information
}}
</output_format>

<important>
- Be thorough - missing a person could lead to compliance failures
- If uncertain about details, include them with lower confidence
- Preserve original spelling and capitalization
- Include nicknames if mentioned (e.g., "Jim (James) Smith")
</important>"""


# =============================================================================
# Name Matching
# =============================================================================

NAME_MATCHING_SYSTEM_PROMPT = """You are an expert at determining if two people are the same individual for financial compliance screening.

You understand:
- Name variations (nicknames, initials, middle names, cultural conventions)
- Age calculation and verification
- The critical importance of not missing true matches in compliance contexts"""

NAME_MATCHING_USER_PROMPT = """<task>
Determine if the query person and the article entity refer to the same individual.
</task>

<query_person>
<name>{query_name}</name>
<date_of_birth>{query_dob}</date_of_birth>
</query_person>

<article_entity>
{entity_json}
</article_entity>

<article_context>
<publish_date>{article_date}</publish_date>
<source>{article_source}</source>
<relevant_snippet>
{context_snippet}
</relevant_snippet>
</article_context>

<matching_considerations>
1. NAME VARIATIONS:
   - Nicknames (James→Jim, Robert→Bob, Richard→Dick, William→Bill, etc.)
   - Initials (J. Smith vs John Smith vs John Q. Smith)
   - Middle names used as first names (John Michael Smith → Michael Smith)
   - Cultural naming conventions:
     * Chinese: surname first (Wang Wei vs Wei Wang)
     * Spanish: double surnames (García López)
     * Arabic: patronymic names (Mohammed bin Salman)
     * Single names (Indonesian, Brazilian)
   - Titles (Dr., Sir, Professor) should be ignored for matching
   - Married vs maiden names for women

2. AGE/DOB VERIFICATION:
   - Calculate expected age from DOB as of article publish date
   - Allow tolerance of ±2 years for exact age
   - For approximate ages ("in his 40s"), verify DOB falls in that range
   - Missing age in article reduces confidence but doesn't eliminate match

3. OTHER IDENTIFIERS:
   - Occupation/job title match increases confidence
   - Location match increases confidence
   - Any contradicting details (e.g., completely different occupation in different field) reduces confidence

4. COMPLIANCE CONTEXT:
   - False negatives (missing a true match) are WORSE than false positives
   - When uncertain, classify as potential match for manual review
   - Only classify as "no match" when clearly different people
</matching_considerations>

<output_format>
Return a JSON object with:
{{
  "is_match": true or false (true if confident match OR uncertain),
  "confidence": "HIGH" / "MEDIUM" / "LOW",
  "match_probability": 0.0 to 1.0 (numeric confidence score),
  "reasoning_steps": [
    "Step 1: Compare names...",
    "Step 2: Verify age...",
    "Step 3: Check other identifiers..."
  ],
  "supporting_evidence": [
    "Name matches with common nickname variation",
    "Age aligns with DOB (45 years old in 2024, DOB 1978)"
  ],
  "contradicting_evidence": [
    "Article mentions location X, query person from location Y"
  ],
  "missing_information": [
    "middle_name",
    "exact_dob_in_article"
  ]
}}
</output_format>

<decision_guidelines>
- match_probability > 0.80 AND clear evidence → is_match=true, confidence=HIGH
- match_probability 0.60-0.80 → is_match=true, confidence=MEDIUM
- match_probability 0.40-0.59 → is_match=true, confidence=LOW (flag for manual review)
- match_probability < 0.40 AND clear contradictions → is_match=false
</decision_guidelines>

<critical>
Think step-by-step. Show your reasoning clearly. In compliance contexts, we CANNOT miss true matches.
</critical>"""


# =============================================================================
# Sentiment Analysis
# =============================================================================

SENTIMENT_ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing whether news articles contain adverse media (negative coverage) about individuals for financial compliance screening.

You understand:
- Adverse media indicators in compliance contexts
- The difference between neutral reporting and negative portrayal
- How to extract specific evidence of adverse information"""

SENTIMENT_ANALYSIS_USER_PROMPT = """<task>
Analyze whether this news article portrays the specified individual in a negative light (adverse media).
</task>

<person_of_interest>
<name>{person_name}</name>
</person_of_interest>

<article>
<title>{article_title}</title>
<source>{article_source}</source>
<publish_date>{publish_date}</publish_date>
<content>
{article_content}
</content>
</article>

<adverse_media_indicators>
Assess whether the article contains negative information in these categories:

1. LEGAL ISSUES:
   - Criminal charges, arrests, indictments
   - Lawsuits (as defendant)
   - Convictions or guilty pleas
   - Ongoing investigations
   - Regulatory enforcement actions

2. FINANCIAL MISCONDUCT:
   - Fraud or embezzlement
   - Money laundering
   - Bankruptcy or insolvency
   - Tax evasion
   - Financial mismanagement

3. ETHICAL VIOLATIONS:
   - Corruption or bribery
   - Conflicts of interest
   - Insider trading
   - Professional misconduct

4. SANCTIONS & RESTRICTIONS:
   - Government sanctions
   - Industry bans or restrictions
   - Travel restrictions
   - Asset freezes

5. REPUTATIONAL DAMAGE:
   - Major scandals or controversies
   - Credible allegations of wrongdoing
   - Loss of professional licenses
   - Forced resignations
</adverse_media_indicators>

<important_distinctions>
- "Was acquitted" / "charges dropped" → NEUTRAL or POSITIVE (resolved favorably)
- "Facing allegations" / "under investigation" → NEGATIVE (adverse media)
- "Witness in case" → NEUTRAL (not accused)
- "Donated to charity" → POSITIVE (unless part of scandal)
- Neutral reporting of facts → evaluate overall context and implications
- Historical negative events that were resolved → still NEGATIVE but note resolution
</important_distinctions>

<output_format>
Return a JSON object with:
{{
  "classification": "POSITIVE" / "NEGATIVE" / "NEUTRAL",
  "is_adverse_media": true or false,
  "severity": "HIGH" / "MEDIUM" / "LOW" (if adverse, otherwise null),
  "adverse_indicators": ["fraud", "lawsuit", "investigation"],
  "evidence_snippets": [
    "Quote or paraphrase from article showing adverse info",
    "Another relevant quote"
  ],
  "positive_indicators": ["charges_dropped", "acquitted"] (if any),
  "context": "Brief explanation of the overall portrayal",
  "reasoning": "Clear explanation of classification decision"
}}
</output_format>

<severity_guidelines>
- HIGH: Criminal convictions, major fraud, sanctions, active money laundering
- MEDIUM: Ongoing investigations, civil lawsuits, regulatory actions, allegations
- LOW: Minor violations, resolved issues, peripheral involvement
</severity_guidelines>

<critical>
Focus on facts presented in the article. Consider both the nature of the allegations/actions AND their current status (active, resolved, alleged, proven).
</critical>"""


# =============================================================================
# Report Generation
# =============================================================================

REPORT_GENERATION_SYSTEM_PROMPT = """You are an expert at creating clear, professional adverse media screening reports for compliance analysts.

Your reports are:
- Concise yet comprehensive
- Well-structured and easy to scan
- Objective and evidence-based
- Actionable with clear recommendations"""

REPORT_GENERATION_USER_PROMPT = """<task>
Generate a professional adverse media screening report based on the analysis results.
</task>

<analysis_results>
{results_json}
</analysis_results>

<report_requirements>
1. Use clear, professional language suitable for compliance officers
2. Lead with the most important information (decision and confidence)
3. Provide specific evidence, not vague statements
4. Highlight any uncertainties or missing information
5. Include actionable recommendations
6. Format for readability (use sections, bullet points where appropriate)
</report_requirements>

<report_structure>
# ADVERSE MEDIA SCREENING REPORT

## QUERY INFORMATION
- Name: [query name]
- Date of Birth: [DOB]
- Article: [title and URL]
- Source: [publication]
- Date: [publish date]
- Screening Date: [timestamp]

## SCREENING DECISION
[Clear statement: MATCH / NO MATCH / UNCERTAIN]
[One paragraph summary of decision and key reasoning]

## MATCH ASSESSMENT
**Confidence Level:** [HIGH / MEDIUM / LOW]
**Match Probability:** [0.XX]

**Key Evidence:**
[Bullet points of supporting evidence]

**Reasoning:**
[Clear explanation of how the match determination was made]

**Missing Information:**
[List any information that would help confirm/deny the match]

## SENTIMENT ANALYSIS
[Only include this section if a match was found]

**Classification:** [POSITIVE / NEGATIVE / NEUTRAL]
**Adverse Media:** [Yes/No]
[If adverse:] **Severity:** [HIGH / MEDIUM / LOW]

**Findings:**
[Bullet points describing adverse media indicators found]

**Evidence:**
[Specific quotes or descriptions from article]

## ENTITIES IDENTIFIED IN ARTICLE
[List all people found in the article with brief context]

## RECOMMENDATION
[Clear next steps: "Proceed with onboarding" / "Requires manual review" / "Escalate to compliance team" / "Reject application"]

[If uncertain or medium/low confidence:] Additional steps suggested:
- [e.g., "Search for additional articles about this individual"]
- [e.g., "Verify middle name through other sources"]
- [e.g., "Request additional documentation from applicant"]

## PROCESSING DETAILS
- Model: [model name]
- Provider: [provider]
- Processing Time: [duration]
- Total Cost: $[cost]

---
*This report was generated by an AI-powered screening system. Manual review is recommended for all uncertain or adverse findings.*
</report_structure>

<tone_guidelines>
- Professional and objective
- Direct and clear (avoid hedging unless genuinely uncertain)
- Evidence-based (cite specific facts from article)
- Actionable (clear on what analyst should do next)
- Balanced (present both supporting and contradicting evidence)
</tone_guidelines>

<critical>
This report will be used for compliance decisions. Ensure all claims are supported by evidence from the analysis. Be clear about limitations and uncertainties.
</critical>"""


# =============================================================================
# Helper Functions
# =============================================================================

def format_entity_for_prompt(entity: dict) -> str:
    """
    Format entity dictionary as XML for inclusion in prompts.

    Args:
        entity: Entity dictionary from extraction

    Returns:
        XML-formatted entity string
    """
    xml = "<entity>\n"
    xml += f"  <full_name>{entity.get('full_name', 'Unknown')}</full_name>\n"

    if entity.get('age'):
        xml += f"  <age>{entity['age']}</age>\n"
    if entity.get('approximate_age_range'):
        xml += f"  <approximate_age_range>{entity['approximate_age_range']}</approximate_age_range>\n"
    if entity.get('occupation'):
        xml += f"  <occupation>{entity['occupation']}</occupation>\n"
    if entity.get('location'):
        xml += f"  <location>{entity['location']}</location>\n"

    if entity.get('other_details'):
        xml += "  <other_details>\n"
        for detail in entity['other_details']:
            xml += f"    <detail>{detail}</detail>\n"
        xml += "  </other_details>\n"

    if entity.get('context_snippet'):
        xml += f"  <context_snippet>{entity['context_snippet']}</context_snippet>\n"

    xml += "</entity>"
    return xml