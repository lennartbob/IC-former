import os
import time
from pathlib import Path
from openai import AzureOpenAI

# --- Configuration ---
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_ENDPOINT = "https://openai-pinkdotai.openai.azure.com/"
API_VERSION = "2025-01-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-2"  # Make sure this is a Global-Batch deployment
BATCH_INPUT_DIR = Path("data/question_batches")
BATCH_OUTPUT_DIR = Path("data/question_batches_output")

# Ensure output directory exists
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# --- Helper Functions ---
def upload_file(file_path: Path):
    with file_path.open("rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    return uploaded_file.id

def create_batch(file_id):
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    return response.id

def wait_for_completion(batch_id):
    print(f"Waiting for batch {batch_id} to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in {"completed", "failed", "cancelled"}:
            print(f"Batch {batch_id} finished with status: {batch.status}")
            return batch
        print(f"Status: {batch.status} - Waiting...")
        time.sleep(30)

def download_file(file_id, save_path: Path):
    content = client.files.download(file_id)
    with open(save_path, "wb") as f:
        f.write(content)

# --- Main Process ---
for file_path in BATCH_INPUT_DIR.glob("*.jsonl"):
    print(f"Processing: {file_path.name}")

    try:
        # Upload file
        file_id = upload_file(file_path)

        # Create batch job
        batch_id = create_batch(file_id)

        # Wait for completion
        batch = wait_for_completion(batch_id)

        # Prepare output paths
        output_base = BATCH_OUTPUT_DIR / file_path.stem
        output_base.mkdir(exist_ok=True)

        # Download results if available
        if batch.output_file_id:
            download_file(batch.output_file_id, output_base / "output.jsonl")
        if batch.error_file_id:
            download_file(batch.error_file_id, output_base / "errors.jsonl")

    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")

print("✅ All batch jobs completed.")
