import json
from pprint import pprint

path = "data/collected_pdf_texts_with_questions.json"

with open(path, encoding="utf-8") as f:
    main_db = json.loads(f.read())


pprint(main_db[0]["questions"])

print("--")

pprint(main_db[3]["questions"])