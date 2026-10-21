
import csv
from adapted_interface import generate_interactive, GenerationConfig

# Model and generation settings
MODEL = "qwen2.5:1.5b"

config = GenerationConfig(
    max_length=30,
    temperature=0.0,
    top_p=1.0,
    do_sample=False,
    repetition_penalty=1.0
)

# Read the original 30 test queries
with open("test.csv", newline="", encoding="utf-8-sig") as f:
    test_data = list(csv.DictReader(f))

# Read the original 6 few-shot examples
with open("examples.csv", newline="", encoding="utf-8-sig") as f:
    examples = list(csv.DictReader(f))

def build_revised_prompt(query):
    prompt = """You are a biological query classification system.

Your task is to identify the PRIMARY INTENT of the CURRENT query.

CLASSIFICATION RULES:

1. PaperSearch
Choose this ONLY when the user requests scientific papers,
research articles, publications, or literature searches.

2. ProteinSearch
Choose this ONLY when the user requests protein sequences,
3D structures, PDB entries, UniProt records, or protein data.

3. GeneralQuestion
Choose this when the user asks for an explanation,
definition, comparison, or biological concept without
requesting external papers or protein database records.

IMPORTANT:
Classify the CURRENT query, not the examples.
Do not assume that every query belongs to PaperSearch.
Return EXACTLY ONE label:
PaperSearch, ProteinSearch, or GeneralQuestion.

LABELED EXAMPLES:
"""

    for row in examples:
        prompt += (
            f"\nQuery: {row['query']}\n"
            f"Correct category: {row['gold_label']}\n"
        )

    prompt += f"""

END OF EXAMPLES.

NOW CLASSIFY THIS NEW QUERY:

Query: {query}

Category:"""

    return prompt


results = []
correct_count = 0

for i, row in enumerate(test_data, start=1):
    query = row["query"]
    gold = row["gold_label"]

    prompt = build_revised_prompt(query)

    prediction = generate_interactive(
        prompt,
        generation_config=config,
        model_name=MODEL
    ).strip()

    is_correct = prediction == gold
    correct_count += int(is_correct)

    results.append({
        "method": "Revised Few-shot",
        "query": query,
        "gold_label": gold,
        "prediction": prediction,
        "correct": is_correct
    })

    print(f"{i}/{len(test_data)} | Gold: {gold} | Predicted: {prediction}")

accuracy = correct_count / len(test_data) * 100

print("\n====================================")
print(f"Revised Few-shot Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct_count}/{len(test_data)}")
print("====================================")

# Save results in a separate file
with open("results_revised.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["method", "query", "gold_label", "prediction", "correct"]
    )
    writer.writeheader()
    writer.writerows(results)

# Save the revised full prompt template
with open("prompt_revised.txt", "w", encoding="utf-8") as f:
    f.write(build_revised_prompt("{query}"))

print("Saved: results_revised.csv")
print("Saved: prompt_revised.txt")
