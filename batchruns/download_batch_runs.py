

import asyncio
import os
from pathlib import Path

from openai import AzureOpenAI
from batchruns.run_folder import download_file
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
async def download_batch_run(batch_ids: list[str], BATCH_OUTPUT_DIR: str, client):
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    for id in batch_ids:
        try:
            # Download file content (assumes this returns an async stream or bytes-like object)
            response = await client.files.content(id)

            # Define the output path
            output_path = os.path.join(BATCH_OUTPUT_DIR, f"{id}.jsonl")

            # Save file asynchronously
            async with aiofiles.open(output_path, "wb") as f:
                content = await response.read()
                await f.write(content)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed to download file {id}: {e}")
if __name__ == "__main__":
    BATCH_OUTPUT_DIR = Path("data/question_batches_output")

    ids = [
        "batch_d0a3e94e-2a0f-486a-83bf-2d3548b97892",
        "batch_0cf0ebbc-110b-4e8c-a600-b6325bc11404",
        "batch_a756b163-2c07-4425-9a2d-85bed1057398",
        "batch_062f6ed7-f7a3-4a7c-ac4d-db1b767bf2d2",
        "batch_1d9b01d3-91fe-46bd-9916-adcbf884f660",
    ]
    asyncio.run(download_batch_run(ids, BATCH_OUTPUT_DIR, client))