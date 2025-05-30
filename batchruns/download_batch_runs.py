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
        "file-4eae347c-de4f-45ff-8c44-4c93c8ca9599",
        "file-93a14d98-28e2-49f0-ad1c-751328da2215",
        "file-e770e8e2-0f92-4c02-b4c3-8e68012f7c0b",
        "file-4f9c9886-2434-4a11-a3e1-f273321a0547"
    ]
    download_batch_run(ids, BATCH_OUTPUT_DIR, client)