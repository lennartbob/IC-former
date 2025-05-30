import json
from collections import defaultdict

file_path = "data/collected_pdf_texts_merged.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the previous script ran successfully.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{file_path}'. Please check the file format.")
    exit()

language_counts = defaultdict(int)
total_documents = len(data)

if total_documents == 0:
    print("No documents found in the dataset.")
else:
    for item in data:
        language = item.get("language")
        if language:  # Ensure 'language' key exists and is not None/empty
            language_counts[language] += 1
        else:
            language_counts["unknown"] += 1 # Handle cases where language might be missing

    print("Language Occurrences and Percentages:")
    print("-" * 50)

    for lang, count in language_counts.items():
        percentage = (count / total_documents) * 100
        print(f"Language: {lang:<15} Count: {count:<8} Percentage: {percentage:.2f}%")

    print("-" * 50)
    print(f"Total documents processed: {total_documents}")