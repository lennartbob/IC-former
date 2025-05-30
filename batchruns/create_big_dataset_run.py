import json
from pprint import pprint

from tqdm import tqdm

from core.bacher import AzureBatchJsonlGenerator
from core.jinja_helper import process_template

def map_langdetect_to_full_language(langdetect_output):
    """
    Maps a langdetect output code (e.g., "en", "nl") to the full language name.
    If the code is not found, it returns "english".

    Args:
        langdetect_output (str): The language code string from langdetect.

    Returns:
        str: The full language name.
    """
    language_map = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "et": "Estonian",
        "fa": "Persian",
        "fi": "Finnish",
        "fr": "French",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "kn": "Kannada",
        "ko": "Korean",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mr": "Marathi",
        "ne": "Nepali",
        "nl": "Dutch",
        "no": "Norwegian",
        "" "": "Other", # Added for cases where langdetect might return an empty string or 'unk'
        "pa": "Punjabi",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "zu": "Zulu",
        # Add more mappings as needed based on the full range of langdetect outputs
    }

    return language_map.get(langdetect_output, "english")

path = "data/collected_pdf_texts_with_questions.json"

with open(path, encoding="utf-8") as f:
    main_db = json.loads(f.read())

bachter = AzureBatchJsonlGenerator(output_dirpath="data/bigger_bachtes")

f = 0
for pdf_item in tqdm(main_db):
    text = pdf_item["text"]
    if "questions" not in pdf_item:
        f +=1
        continue
    questions = pdf_item["questions"]
    lang = map_langdetect_to_full_language(pdf_item["language"])
    for que in questions:
        if que["answer"] != "":
            continue
        if "q_id" not in que:
            continue
        id = que["q_id"]
        prompt = process_template(
            "contextual_summ_2.jinja",
            {"queries":que["question"], "context":text, "return_language": lang}
        )
        bachter.add_request(
            prompt=prompt,
            format=False,
            custom_id=id
        )

print("PDF items with no questions:",f)
bachter.close_file()



