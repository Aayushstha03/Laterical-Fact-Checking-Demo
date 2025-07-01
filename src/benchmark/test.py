import os
import pandas as pd
import jsonlines

from gemini_test import custom_search_api, verify_claim


# Path to the fever dataset
path = "./train.jsonl"

# Translation dictionary for FEVER labels to local format
fever_label_translation={
    "SUPPORTS": "supported",
    "REFUTES": "refuted",
    "NOT ENOUGH INFO": "Insufficient information",}

# Evaluate a single instance of claim and label
def evaluate_instance(claim, label):
    evaluation_result = False        
    predicted_label = "Insufficient information"
    verification_result = {}

    sources = custom_search_api(claim)

    if sources:
        verification_result = verify_claim(claim, sources)
        predicted_label = verification_result.get("assessment", "Insufficient information")

   
    evaluation_result = predicted_label == fever_label_translation.get(label, "")

    return {
        "claim": claim,
        "label": label,
        "predicted_label": predicted_label,
        "evaluation_result": evaluation_result,
        "verification_result": verification_result
    }


# Run the benchmark on the FEVER dataset and save results to a JSONL file
def run_benchmark(n=100,output_path="fever_eval_results.jsonl"):
    results = []  

    with jsonlines.open(path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for i, item in enumerate(reader):
            # Limit to n instances for testing
            if i >= n:
                break

            # Extract parameters
            claim = item.get("claim", "")
            label = item.get("label", "")

            # Run evaluation
            try:
                result = evaluate_instance(claim, label)
            except Exception as e:
                print(f"[{i+1}] Error evaluating claim: {e}")
                result = {
                    "claim": claim,
                    "label": label,
                    "predicted_label": "ERROR",
                    "evaluation_result": False,
                    "verification_result": {"error": str(e)}
                }

            # Print the result
            print(f"[{i+1}] Claim: {claim}")
            print(f"Label: {label}")
            print(f"Predicted: {result['predicted_label']}")
            print(f"Correct? {result['evaluation_result']}")
            print("-" * 50)

            results.append(result)
            writer.write(result)  # Write each result to JSONL file
    
    
    correct = sum(1 for r in results if r["evaluation_result"])
    print(f"\nAccuracy: {correct}/{len(results)} = {correct / len(results):.2%}")

    return results
      
                     

run_benchmark(100)

           



   
    
