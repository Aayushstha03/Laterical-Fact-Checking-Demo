import json
import jsonlines

from gemini_test import custom_search_api, verify_claim


# Translation dictionary for mapping AVERITEC labels to verification output labels
label_translation = {
    "Supported": "supported",
    "Supports":"supported",
    "Refuted": "refuted",
    "Refutes":"refuted",    
    "Not Enough Evidence": "Insufficient information",
    }

# Evaluate a single claim
def evaluate_instance(claim, label):
         
    predicted_label = "Insufficient information"
    verification_result = {}

    sources = custom_search_api(claim)

    if sources:
        verification_result = verify_claim(claim, sources)
        predicted_label = verification_result.get("assessment", "Insufficient information")

    
    evaluation_result = predicted_label == label_translation.get(label,"")

    return {
        "claim": claim,
        "label": label,
        "predicted_label": predicted_label,
        "evaluation_result": evaluation_result,
        "verification_result": verification_result
    }


# Run benchmark on the dataset
def run_benchmark(n=100, output_path="new_eval_results.jsonl",dataset_path="./data/averitec_dev.json"):
   
    results = []

    # Read JSON array from the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

 
    with jsonlines.open(output_path, mode='w') as writer:
        for i, item in enumerate(data):
            # Limit the number of instances processed
            if i >= n:
                break

            # Extract parameters from the item
            claim = item.get("claim", "")
            label = item.get("label", "")

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

            print(f"[{i+1}] Claim: {claim}")
            print(f"Label: {label}")
            print(f"Predicted: {result['predicted_label']}")
            print(f"Correct? {result['evaluation_result']}")
            print("-" * 50)

            result["label_justification"]=item.get("justification", "")
            results.append(result)
            writer.write(result)

    # Accuracy Summary
    correct = sum(1 for r in results if r["evaluation_result"])
    print(f"\nAccuracy: {correct}/{len(results)} = {correct / len(results):.2%}")

    return results

# Run the benchmark
# run_benchmark(100,"averitec_eval_results_1.jsonl")
