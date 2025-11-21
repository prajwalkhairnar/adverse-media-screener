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

ENTITY_EXTRACTION_USER_PROMPT = f"""<task>
Extract all people mentioned in the following news article. For each person, identify any available details that could help verify their identity.
</task>

<article>
<url>{{article_url}}</url>
<title>{{article_title}}</title>
<source>{{article_source}}</source>
<publish_date>{{publish_date}}</publish_date>
<language>{{language}}</language>
<content>
{{article_content}}
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
Return a JSON object matching the ExtractionOutput schema.
The object must contain a key 'extracted_entities', which holds the list of person objects:

<extracted_entities>
  <person>
    <full_name>exact name as in article</full_name>
    <age type="numeric">numeric age if mentioned (or null)</age>
    <approximate_age_range>description like 'in his 40s' (or null)</approximate_age_range>
    <occupation>job title or role (or null)</occupation>
    <location>city, country, or region (or null)</location>
    <other_details type=list of string facts>
      <fact>factual detail 1 (MUST NOT contain inner tags)</fact>
      <fact>factual detail 2 (MUST NOT contain inner tags)</fact>
    </other_details>
    <context_snippet>1-2 sentences showing how person is described</context_snippet>
    <confidence>high/medium/low based on clarity of information</confidence>
  </person>
</extracted_entities>

</output_format>

<important>
- Be thorough - missing a person could lead to compliance failures
- If uncertain about details, include them with lower confidence
- Preserve original spelling and capitalization
- Include nicknames if mentioned (e.g., "Jim (James) Smith")
</important>

<critical>
When generating the final JSON output, ensure all string values (especially names and snippets) do not contain unnecessary escape characters (like backslashes before apostrophes: \\'). Output the cleanest possible JSON.
</critical>
"""


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
{entity_xml}
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

Return a JSON object that strictly adheres to the NameMatchingOutput schema.
The object must a single top-level key: "final_assessment", which holds the rest:

<final_assessment>
  <is_match>true or false</is_match>
  <confidence>HIGH/MEDIUM/LOW</confidence>
  <match_probability>0.0 to 1.0 (numeric confidence score)</match_probability>
  <reasoning_steps>
    <step>Step 1: Compare names...</step>
    <step>Step 2: Verify age...</step>
  </reasoning_steps>
  <supporting_evidence>
    <evidence>Name matches with common nickname variation</evidence>
    <evidence>Age aligns with DOB (45 years old in 2024, DOB 1978)</evidence>
  </supporting_evidence>
  <contradicting_evidence>
    <evidence>Article mentions location X, query person from location Y</evidence>
  </contradicting_evidence>
  <missing_information>
    <item>middle_name</item>
    <item>exact_dob_in_article</item>
  </missing_information>
</final_assessment>

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
{article_text}
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

<output_format type=JSON>

Return a JSON object that strictly adheres to the SentimentOutput schema.
The object must a single top-level key: "assessment", which holds the rest:
Should strictly be a valid JSON, see below structure for reference. 

<assessment>
  <classification>NEGATIVE</classification>
  <is_adverse_media>true</is_adverse_media>
  <severity>HIGH</severity>
  <adverse_indicators>
    <indicator>securities fraud</indicator>
    <indicator>hate speech</indicator>
  </adverse_indicators>
  <evidence_snippets>
    <snippet>"Musk had not secured the necessary financial backing..."</snippet>
    <snippet>"Antisemitic and racist tweets spiked..."</snippet>
  </evidence_snippets>
  <reasoning>This article details...</reasoning>
</assessment>

</output_format>


<severity_guidelines>
- HIGH: Criminal convictions, major fraud, sanctions, active money laundering
- MEDIUM: Ongoing investigations, civil lawsuits, regulatory actions, allegations
- LOW: Minor violations, resolved issues, peripheral involvement
</severity_guidelines>

<critical>
Focus on facts presented in the article. Consider both the nature of the allegations/actions AND their current status (active, resolved, alleged, proven).
</critical>
"""


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
    xml += f"  <full_name>{entity.full_name}</full_name>\n"

    if entity.age is not None:
        xml += f"  <age>{entity.age}</age>\n"
    if entity.approximate_age_range: # Pydantic treats empty string/list as falsey
        xml += f"  <approximate_age_range>{entity.approximate_age_range}</approximate_age_range>\n"
    if entity.occupation:
        xml += f"  <occupation>{entity.occupation}</occupation>\n"
    if entity.location:
        xml += f"  <location>{entity.location}</location>\n"

    if entity.other_details:
        xml += "  <other_details>\n"
        # FIX 3: Iterate directly over the Pydantic list field
        for detail in entity.other_details:
            xml += f"    <detail>{detail}</detail>\n"
        xml += "  </other_details>\n"

    # context_snippet is a required string field
    if entity.context_snippet:
        xml += f"  <context_snippet>{entity.context_snippet}</context_snippet>\n"

    xml += "</entity>"

    return xml