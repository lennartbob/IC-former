import json

path = "data/collected_pdf_texts_with_questions.json"

with open(path, encoding="utf-8") as f:
    main_db = json.loads(f.read())


print(main_db[0])

print("--")

print(main_db[3])