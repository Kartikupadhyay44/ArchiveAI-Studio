"""
ArchiveAI — Modern History & Archival Chronicle Assistant
Test Suite: Official Assessment Questions, Paraphrased Checks, and Grounding Safeguards
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from rag import answer

OFFICIAL_ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What exact annual guaranteed interest rate, expressed as a percentage, "
            "did the colonial administration promise private British railway companies "
            "under the 1849 contract terms?"
        ),
        "expected_answer": "5% per annum",
        "expected_sources": ["page_13"],
        "expected_status": "SUPPORTED",
    },
    {
        "id": "Q2",
        "question": (
            "Which engineering obstacle on the Bhor Ghat incline made the reversing "
            "station described in Chapter 3 necessary, and what was the gradient ratio?"
        ),
        "expected_answer": (
            "The sheer basalt precipice of Battery Hill above the Khandala ravine; "
            "the ruling gradient was 1 in 37."
        ),
        "expected_sources": ["page_26"],
        "expected_status": "SUPPORTED",
    },
    {
        "id": "Q3",
        "question": (
            "What is the ticket price for an Executive Chair Car seat on the "
            "Vande Bharat Express between New Delhi and Varanasi?"
        ),
        "expected_answer": "[not found]",
        "expected_sources": [],
        "expected_status": "NOT_FOUND",
    },
    {
        "id": "Q4",
        "question": (
            "According to the archival records in the text, who supervised the laying of "
            "the experimental Diamond Harbour underwater telegraph cable in 1851?"
        ),
        "expected_answer": "Dr. William Brooke O'Shaughnessy, Assay Master of the Calcutta Mint",
        "expected_sources": ["page_43"],
        "expected_status": "SUPPORTED",
    },
    {
        "id": "Q5",
        "question": "Which country won the 2022 FIFA World Cup?",
        "expected_answer": "[not found]",
        "expected_sources": [],
        "expected_status": "NOT_FOUND",
    },
]

ADDITIONAL_CONTROL_TESTS = [
    {
        "id": "P1 (Paraphrased Q1)",
        "question": "Under the 1849 contract, what guaranteed interest rate per year was promised to private British railway firms?",
        "expected_substring": "5% per annum",
        "expected_sources": ["page_13"],
    },
    {
        "id": "P2 (Paraphrased Q2)",
        "question": "What engineering obstacle at Battery Hill was faced during the Bhor Ghat incline construction, and what gradient was negotiated?",
        "expected_substring": "1 in 37",
        "expected_sources": ["page_26"],
    },
    {
        "id": "P3 (Paraphrased Q4)",
        "question": "Who was the supervisor for laying the 1851 Diamond Harbour telegraph line?",
        "expected_substring": "Dr. William Brooke O'Shaughnessy",
        "expected_sources": ["page_43"],
    },
    {
        "id": "U1 (Unsupported Control)",
        "question": "What was the capital of France in 1850?",
        "expected_substring": "[not found]",
        "expected_sources": [],
    },
    {
        "id": "U2 (Unsupported Control)",
        "question": "Who won the Indian Premier League in 2024?",
        "expected_substring": "[not found]",
        "expected_sources": [],
    },
    {
        "id": "U3 (Unsupported Control)",
        "question": "What is the speed of sound in dry air at 20 degrees Celsius?",
        "expected_substring": "[not found]",
        "expected_sources": [],
    },
]


def run_official_tests() -> List[Dict[str, Any]]:
    """Execute the 5 official assessment questions and evaluate results."""
    print("=" * 78)
    print("ArchiveAI Assessment Suite: Running 5 Official Assessment Questions")
    print("=" * 78)

    results = []
    all_passed = True

    for test in OFFICIAL_ASSESSMENT_QUESTIONS:
        t0 = time.time()
        res = answer(test["question"])
        elapsed = time.time() - t0

        actual_ans = res.get("answer", "")
        actual_src = res.get("sources", [])

        # Normalize comparison
        expected_ans = test["expected_answer"]
        expected_src = test["expected_sources"]

        if expected_ans == "[not found]":
            ans_match = actual_ans == "[not found]"
            src_match = actual_src == []
        else:
            ans_match = expected_ans.lower() in actual_ans.lower() or actual_ans.lower() in expected_ans.lower()
            src_match = actual_src == expected_src

        passed = ans_match and src_match
        if not passed:
            all_passed = False

        status_str = "PASS [✓]" if passed else "FAIL [✗]"

        print(f"\n{test['id']}: {test['question']}")
        print(f"  Status:          {status_str} ({elapsed:.3f}s)")
        print(f"  Expected Answer: {expected_ans}")
        print(f"  Actual Answer:   {actual_ans}")
        print(f"  Expected Sources:{expected_src}")
        print(f"  Actual Sources:  {actual_src}")

        results.append({
            "id": test["id"],
            "question": test["question"],
            "expected_answer": expected_ans,
            "actual_answer": actual_ans,
            "expected_sources": expected_src,
            "actual_sources": actual_src,
            "answer_matched": ans_match,
            "sources_matched": src_match,
            "passed": passed,
            "elapsed_seconds": round(elapsed, 4),
        })

    return results, all_passed


def run_control_tests() -> bool:
    """Execute additional paraphrased and unsupported control tests."""
    print("\n" + "=" * 78)
    print("ArchiveAI Control Suite: Paraphrased & Unsupported Control Tests")
    print("=" * 78)

    all_passed = True
    for test in ADDITIONAL_CONTROL_TESTS:
        t0 = time.time()
        res = answer(test["question"])
        elapsed = time.time() - t0

        actual_ans = res.get("answer", "")
        actual_src = res.get("sources", [])
        expected_sub = test["expected_substring"]
        expected_src = test["expected_sources"]

        if expected_sub == "[not found]":
            passed = (actual_ans == "[not found]") and (actual_src == [])
        else:
            passed = (expected_sub.lower() in actual_ans.lower()) and (actual_src == expected_src)

        if not passed:
            all_passed = False

        status_str = "PASS [✓]" if passed else "FAIL [✗]"
        print(f"\n{test['id']}: {test['question']}")
        print(f"  Status: {status_str} | Answer: {actual_ans} | Sources: {actual_src}")

    return all_passed


def main():
    official_results, official_passed = run_official_tests()
    control_passed = run_control_tests()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_results": official_results,
        "official_all_passed": official_passed,
        "control_all_passed": control_passed,
        "summary": {
            "total_official_tests": len(official_results),
            "passed_official_tests": sum(1 for r in official_results if r["passed"]),
            "failed_official_tests": sum(1 for r in official_results if not r["passed"]),
            "success_rate": f"{(sum(1 for r in official_results if r['passed']) / len(official_results)) * 100:.1f}%",
        },
    }

    report_path = Path(__file__).resolve().parent / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Test Execution Summary: {report['summary']['passed_official_tests']}/{report['summary']['total_official_tests']} Passed ({report['summary']['success_rate']})")
    print(f"Full evaluation report saved to: {report_path}")
    print("=" * 78)

    if not official_passed or not control_passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
