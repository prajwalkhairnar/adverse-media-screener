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
        "url": "https://www.microsoft.com/en-us/exec/satya-nadella",  # Official bio page or neutral/positive profile
        "description": "Non-adverse outcome; CEO biography with positive, neutral content.",
        "expected_decision": "NO_MATCH"
    },
    {
        "name": "John Smith",
        "dob": "1990-01-01",
        "url": "https://www.bbc.com/news/uk-england-london-65996068",  # Article about arrest of a John Smith with minimal identifying info
        "description": "Ambiguous match; common name with adverse event but no unique identifiers to confirm individual.",
        "expected_decision": "UNCERTAIN"
    },
    {
        "name": "Paul Anderson",
        "dob": "1978-01-01",
        "url": "https://www.bbc.co.uk/news/business-62812345",  # Article about major financial scandal unrelated to Paul Anderson
        "description": "Match-negative no client mention; article about adverse event, client name not mentioned so no match.",
        "expected_decision": "NO_MATCH"
    }
]





def run_test_case(case: dict):
    # Base command: python -m src.main screen
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
        capture_output=False,  # Set to True if you want to capture output instead of printing it directly
        text=True,
        check=False  # Don't raise error if the script itself fails, handle manually
    )

    if result.returncode == 0:
        print(f"✅ Case {case['name']} COMPLETED successfully.")
    else:
        print(f"❌ Case {case['name']} FAILED with return code {result.returncode}.")
        # Optional: Print error output if captured
        # print(result.stderr) 


if __name__ == "__main__":
    print("Starting End-to-End Test Suite...")
    for i, case in enumerate(TEST_CASES):
        run_test_case(case)
    print("\nEnd-to-End Test Suite FINISHED.")