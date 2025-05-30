import json
from uuid import uuid4

from core.bacher import AzureBatchJsonlGenerator
from core.jinja_helper import process_template

path = "data/collected_pdf_texts_async.json"


with open(path, encoding="utf-8") as f:
    data =json.loads(f.read())

bachter = AzureBatchJsonlGenerator(output_dirpath="data/question_batches")
for pdf_item in data:
    pdf_id = pdf_item["filename"].split(".")[0]
    prompt = process_template(
        "question_gen.jinja",
        {"context":pdf_item["text"]}
    )
    bachter.add_request(
        prompt=prompt,
        format=True,
        custom_id=pdf_id
    )

bachter.close_file()

    


