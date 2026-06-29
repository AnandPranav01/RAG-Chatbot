
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.generate import answer_question

TEST_CASES = [
    # Clearly answerable from documents
    "What is the employee leave policy?",
    "How many paid leaves do employees get annually?",
    "What is the refund window for purchases?",
    "What happens if a product arrives damaged?",
    "Is multi-factor authentication mandatory?",
    "How long is the probation period for new employees?",
    "What are the subscription plans for Anthrasync Cloud Suite?",
    "How often is user data backed up?",

    "What is the leave policy?",  # could mean several leave types

  
    "What is the capital of France?",
    "What is the company's annual revenue?",
]


def run_tests():
    for i, question in enumerate(TEST_CASES, start=1):
        print(f"\n--- Test {i}: {question} ---")
        result = answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    run_tests()
