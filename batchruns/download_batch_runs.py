import os
from pathlib import Path

from openai import AzureOpenAI
import aiofiles

API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-pinkdotai.openai.azure.com/")
API_VERSION = "2025-01-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-2"  # Make sure this is a Global-Batch deployment

# Validate configuration
if not API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")
if not AZURE_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set.")

# Initialize AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

def download_batch_run(batch_ids: list[str], BATCH_OUTPUT_DIR: str, client):
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    for id in batch_ids:
        try:
            # Download file content
            response = client.files.content(id)

            # Define the output path
            output_path = os.path.join(BATCH_OUTPUT_DIR, f"{id}.jsonl")

            # Save file synchronously
            # The .read() method on the HttpxBinaryResponseContent object is synchronous
            with open(output_path, "wb") as f:
                content = response.read()
                f.write(content)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed to download file {id}: {e}")

if __name__ == "__main__":
    BATCH_OUTPUT_DIR = Path("data/question_batches_output")

    ids = [
        "file-2c083a73-fe51-4ff7-afc4-e8089d8bfbf4",
        "file-b086b7b6-d6d7-4e62-a7ae-98dee71a3431",
        "file-46837b6b-6599-4a33-84f1-5a14cf5d120e",
        "file-512bd224-b010-45ea-9e81-7a9cd003a11b",
        "file-69434a52-c24a-4588-bddc-bc6d07da657e"
    ]
    download_batch_run(ids, BATCH_OUTPUT_DIR, client)