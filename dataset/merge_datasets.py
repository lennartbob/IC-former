import json

old = "data/collected_pdf_texts_with_q_and_answer_id_updated.json"
new = "data/collected_pdf_texts_async.json"
output_file = "data/collected_pdf_texts_merged.json" # New file to save the merged data

with open(old, encoding="utf-8") as f:
    old_data = json.loads(f.read())

with open(new, encoding="utf-8") as f:
    new_data = json.loads(f.read())

# Create a dictionary from old_data for efficient lookup by filename
old_data_dict = {item["filename"]: item for item in old_data}

merged_data_dict = {item["filename"]: item for item in new_data}
merged_data_dict.update(old_data_dict) # This will overwrite entries in merged_data_dict if keys match

# Convert the dictionary back to a list of dictionaries
merged_data = list(merged_data_dict.values())

# Save the merged data to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"Merged data saved to {output_file}")