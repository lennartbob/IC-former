from typing import Any
from uuid import uuid4
from attrs import define
import json
import os

@define
class Answer:
    id: str
    answer: dict[Any]

folder = "data/question_batches_output"
main = "data/collected_pdf_texts_merged.json"

with open(main, encoding="utf-8") as f:
    main_db = json.loads(f.read())

answers: list[Answer] = []
errors = 0

for filename in os.listdir(folder):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(folder, filename)
        print(f"Opening and reading: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                line_data = json.loads(line)

                if not line_data["response"]:
                    print("found an error!")
                    errors += 1
                    continue
                # Assuming 'answer' is a string that needs to be parsed into a dictionary
                # based on the original code's structure, it seems 'answer' itself
                # contains the 'questions' key directly.
                # Let's adjust this line to correctly parse if 'content' is a JSON string.
                try:
                    content_str = line_data["response"]["body"]["choices"][0]["message"]["content"]
                    answer_data = json.loads(content_str) # Parse the content string
                    answers.append(Answer(
                        id=line_data["custom_id"],
                        answer=answer_data["questions"] # Access 'questions' from the parsed data
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing content for {line_data['custom_id']}: {e}")
                    errors += 1
                    continue


print(f"Processed answers! Found {len(answers)}. Errors: {errors}")

def find_db_item_by_pdf_id(main_db, pdf_id):
    """Find the database item that matches the given PDF ID"""
    for item in main_db:
        if "filename" in item:
            # Extract PDF ID from filename (remove .pdf extension)
            item_pdf_id = item["filename"].split(".")[0]
            if item_pdf_id == pdf_id:
                return item
    return None

# Process each answer and add it to the corresponding database item
matched = 0
not_matched = 0

for ans in answers:
    # Extract PDF ID from answer ID (assuming format like "pdf_id_something")
    # You might need to adjust this based on your actual ID format
    try:
        # If your ID format is like "filename_batch_number" or similar
        pdf_id = ans.id.split("_")[0]  # Adjust this split logic as needed

        # Find the matching database item
        db_item = find_db_item_by_pdf_id(main_db, pdf_id)

        if db_item is not None:
            # Add the questions to the database item
            db_item["questions"] = ans.answer
            for ques in db_item["questions"]:
                ques["q_id"] = uuid4().hex
                ques["answer"] = ""
                if ques["type"] == "random":
                    ques["answer"] = "[Nothing relevant found]"
            matched += 1
            print(f"Matched answer for PDF ID: {pdf_id}")
        else:
            print(f"No matching database item found for PDF ID: {pdf_id}")
            not_matched += 1

    except Exception as e:
        print(f"Error processing answer ID {ans.id}: {e}")
        not_matched += 1

print(f"Matching complete! Matched: {matched}, Not matched: {not_matched}")

# Save the updated database
output_file = "data/collected_pdf_texts_with_questions.json"

# Calculate the size of the JSON data before saving
json_data_string = json.dumps(main_db, ensure_ascii=False, indent=2)
print(f"Size of JSON data to be saved: {len(json_data_string.encode('utf-8'))} bytes")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data_string)

print(f"Updated database saved to: {output_file}")
