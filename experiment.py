
import csv
from adapted_interface import (
    GenerationConfig,
    generate_interactive
)

LABELS = [
    "PaperSearch",
    "ProteinSearch",
    "GeneralQuestion"
]

INSTRUCTIONS = """
You are a biological information retrieval assistant.

Classify the user's query into exactly one category.

PaperSearch:
The user wants to find scientific papers,
research articles, or publications.

ProteinSearch:
The user wants to retrieve protein sequences,
structures, or protein database records.

GeneralQuestion:
The user wants a biological explanation
without retrieving external data.

Choose the user's primary intent.

Return only one of these labels:
PaperSearch, ProteinSearch, GeneralQuestion.

Do not explain your answer.
"""

def load_data(filename):
    with open(filename, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)

def build_prompt(query, examples=None):
    prompt = INSTRUCTIONS.strip() + "\n\n"

    if examples:
        prompt += "Examples:\n\n"

        for row in examples:
            prompt += (
                f"Query: {row['query']}\n"
                f"Label: {row['gold_label']}\n\n"
            )

    prompt += f"Query: {query}\nLabel:"
    return prompt

def evaluate(method, test_data, examples=None):
    results = []

    config = GenerationConfig(
        max_length=30,
        temperature=0.0,
        do_sample=False
    )

    for i, row in enumerate(test_data, 1):
        prompt = build_prompt(row["query"], examples)

        prediction = generate_interactive(
            prompt,
            generation_config=config
        )

        gold_label = row["gold_label"]
        correct = prediction == gold_label

        results.append({
            "method": method,
            "query": row["query"],
            "gold_label": gold_label,
            "prediction": prediction,
            "correct": correct
        })

        print(
            f"{method}: {i}/{len(test_data)} "
            f"| Prediction: {prediction} "
            f"| Correct: {correct}",
            flush=True
        )

    accuracy = (
        sum(r["correct"] for r in results)
        / len(results) * 100
    )

    print(f"\n{method} Accuracy: {accuracy:.2f}%\n")

    return results

# Load datasets
test_data = load_data("test.csv")
few_shot_examples = load_data("examples.csv")

# Check dataset size
assert len(test_data) == 30, (
    f"Expected 30 test samples, got {len(test_data)}"
)
assert len(few_shot_examples) == 6, (
    f"Expected 6 few-shot examples, got {len(few_shot_examples)}"
)

# Check labels
for row in test_data + few_shot_examples:
    assert row["gold_label"] in LABELS, (
        f"Invalid label: {row['gold_label']}"
    )

# Check duplicate queries
test_queries = [r["query"] for r in test_data]
example_queries = [r["query"] for r in few_shot_examples]

assert len(set(test_queries)) == len(test_queries)
assert len(set(example_queries)) == len(example_queries)
assert not set(test_queries) & set(example_queries)

# Experiment 1: Zero-shot
zero_results = evaluate(
    "Zero-shot",
    test_data
)

# Experiment 2: Few-shot
few_results = evaluate(
    "Few-shot",
    test_data,
    few_shot_examples
)

# Save results
all_results = zero_results + few_results

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "query",
            "gold_label",
            "prediction",
            "correct"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)

# Save full prompt templates
with open("prompts.txt", "w", encoding="utf-8") as f:
    f.write("Model: qwen2.5:1.5b\n")
    f.write("Temperature: 0\n\n")

    f.write("ZERO-SHOT PROMPT:\n")
    f.write(build_prompt("{query}"))

    f.write("\n\nFEW-SHOT PROMPT:\n")
    f.write(build_prompt("{query}", few_shot_examples))

print("Experiment completed!")
print("Results saved to results.csv")
print("Prompts saved to prompts.txt")

print("\nIncorrect predictions:")
for row in all_results:
    if not row["correct"]:
        print(row)
