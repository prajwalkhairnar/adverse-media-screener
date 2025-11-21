# run_e2e.py

import subprocess


TEST_CASES = [
    {
        "name": "Elizabeth Holmes",
        "dob": "1984-02-03",
        "url": "https://www.bbc.co.uk/news/world-us-canada-63685131",
        "description": "High-risk adverse match; convicted of investor fraud in Theranos case with high severity risk.",
        "expected_decision": "MATCH"
    },
    {
        "name": "Satya Nadella",
        "dob": "1967-08-19",
        "url": "https://www.ft.com/content/718d5bac-8bcb-4aaf-8554-08763b6144e4", 
        "description": "Match-negative no client mention; Article about apple, neutral or slighly negative content.",
        "expected_decision": "NO_MATCH"
    },
    {
        "name": "Chris Smith",
        "dob": "1990-01-01",
        "url": "https://www.skysports.com/football/news/11095/13472276/sheffield-wednesday-vs-sheffield-united-why-the-first-steel-city-derby-of-the-season-carries-more-weight-than-usual",  # Article about arrest of a John Smith with minimal identifying info
        "description": "Ambiguous match; common name with adverse event but no unique identifiers to confirm individual.",
        "expected_decision": "UNCERTAIN"
    },
    {
        "name": "Paul Anderson",
        "dob": "1978-01-01",
        "url": "https://www.bbc.co.uk/news/articles/cjr4z2g5557o",
        "description": "Match-negative no client mention; article about adverse event, client name not mentioned so no match.",
        "expected_decision": "NO_MATCH"
    }
]



# run_e2e.py

def parse_decision(output: str) -> str:
    """
    Parses the final screening decision from the output summary table.
    Looks for the line containing 'Screening Decision'
    """
    for line in output.splitlines():
        # Look for the line in the table that contains the decision:
        # e.g., '│ Screening Decision │ MATCH               │'
        if 'Screening Decision' in line:
            # Split the line by the column separator (│), take the third element, and clean it up.
            parts = line.split('│')
            if len(parts) >= 3:
                # The decision is in the third column (index 2 or 3 depending on leading/trailing columns)
                # Let's target the last non-empty part which contains the decision
                decision = parts[-2].strip() 
                # Decision will be 'MATCH', 'NO_MATCH', or 'UNCERTAIN'
                if decision in ["MATCH", "NO_MATCH", "UNCERTAIN"]:
                    return decision
    return "UNKNOWN" # Returns UNKNOWN if the decision row wasn't found


def run_test_case(case: dict):

    command_parts = [
        "python", "-m", "src.main", "screen",
        f"--name", case["name"],
        f"--dob", case["dob"],
        f"--url", case["url"]
    ]
    
    print(f"\n--- Running Case: {case['name']} ({case['description']}) ---")
    
    # Execute the command
    result = subprocess.run(
        command_parts,
        capture_output=True,
        text=True,
        check=False
    )

    actual_decision = parse_decision(result.stdout)
    
    if result.returncode == 0:
        print(f"✅ Case {case['name']} COMPLETED successfully. Decision: {actual_decision}")
        # Print the output that was captured, which includes the report path.
        print(result.stdout) 

        return {
            "name": case["name"],
            "expected": case["expected_decision"],
            "actual": actual_decision
        }

    else:
        print(f"❌ Case {case['name']} FAILED with return code {result.returncode}.")
        # Print the error stream to see why the script crashed
        print("\n--- ERROR OUTPUT (stderr) ---")
        print(result.stderr) 
        print("-----------------------------")

        return None


def calculate_metrics(results: list):
    """
    Calculates Recall and Precision where both 'MATCH' and 'UNCERTAIN' predictions
    are treated as the positive prediction (i.e., 'Review Required').
    This maximizes sensitivity (Recall) for the true adverse cases.
    """
    
    # 1. Define the classes
    TARGET_ACTUAL_POSITIVE = {"MATCH", "UNCERTAIN"}
    PREDICTED_POSITIVE_CLASSES = {"MATCH", "UNCERTAIN"}

    # 2. Initialize Confusion Matrix Counts
    TP = 0  # True Positives: Actual MATCH predicted as MATCH or UNCERTAIN
    FP = 0  # False Positives: Actual NO_MATCH/UNCERTAIN predicted as MATCH or UNCERTAIN
    FN = 0  # False Negatives: Actual MATCH predicted as NO_MATCH

    for res in results:

        if res is None: 
            continue

        expected = res["expected"]
        actual = res["actual"]
        
        # Check if the model predicted a "positive" outcome (Review Required)
        is_predicted_positive = actual in PREDICTED_POSITIVE_CLASSES
        
        # Check if the actual outcome is a "positive" outcome (True Adverse Case)
        is_actual_positive = expected in TARGET_ACTUAL_POSITIVE

        if is_actual_positive:
            if is_predicted_positive:
                # True Positive (TP): We flagged a true adverse case for review.
                TP += 1
            else:
                # False Negative (FN): We missed a true adverse case (CRITICAL FAILURE).
                FN += 1
        else: # Actual is NO_MATCH or UNCERTAIN
            if is_predicted_positive:
                # False Positive (FP): We flagged a non-adverse case for review (Analyst overhead).
                FP += 1
            # else: True Negative (TN) - Actual NO_MATCH predicted NO_MATCH (Discarded correctly)
    
    # 3. Calculate Metrics
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # 4. Calculate the False Negative Rate (FNR), which is 1 - Recall, for the report
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "FNR": FNR,
        "Recall": recall,
        "Precision": precision,
        "Target Class": "REVIEW_REQUIRED (MATCH or UNCERTAIN)"
    }

if __name__ == "__main__":
    print("Starting End-to-End Test Suite...")
    
    all_results = []
    for i, case in enumerate(TEST_CASES):
        result = run_test_case(case)
        all_results.append(result)

    print("\n" + "="*50)
    print("End-to-End Test Suite FINISHED.")
    print("="*50)

    # --- SENSITIVITY METRICS CALCULATION ---
    sensitive_metrics = calculate_metrics(all_results) 

    print("\n## 📊 System Metrics (High Sensitivity/Recall Focus) 🔎")
    print("---")
    print(f"**Predicted Positive Class:** {sensitive_metrics['Target Class']}")
    print(f"True Positives (TP): {sensitive_metrics['TP']}")
    print(f"False Positives (FP): {sensitive_metrics['FP']}")
    print(f"False Negatives (FN): {sensitive_metrics['FN']} **(Should be 0 to meet assignment constraint)**")
    
    print(f"\n**RECALL (Sensitivity):** {sensitive_metrics['Recall']:.4f}")
    print(f"**FALSE NEGATIVE RATE (FNR):** {sensitive_metrics['FNR']:.4f}")
    print(f"**PRECISION (Quality of Review Pool):** {sensitive_metrics['Precision']:.4f}")
    print("---")