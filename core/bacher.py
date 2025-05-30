from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any, Optional, TextIO, Set
from uuid import uuid4 # Added Set

from PIL import Image

class AzureBatchJsonlGenerator:
    """
    Generates JSONL lines for Azure OpenAI batch processing.
    Each line corresponds to a single chat completion request.
    Automatically splits output into multiple files if max_file_size_bytes is approached.
    """

    DEFAULT_MAX_FILE_SIZE_BYTES = 200_000_000 # Approx 190.7 MiB

    def __init__(
        self,
        output_dirpath: str,
        base_filename: str = "batch_output.jsonl",
        deployment_name: str = "gpt-4.1-2",
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    ):
        """
        Initializes the generator.

        Args:
            output_dirpath (str): The directory where batch .jsonl files will be saved.
            base_filename (str): The base name for the output files (e.g., "requests.jsonl").
                                 Files will be named like "requests_0.jsonl", "requests_1.jsonl".
            deployment_name (str): The name of your Azure OpenAI Global Batch deployment.
                                   This will be used as the 'model' in the batch request body.
            max_file_size_bytes (int): Maximum size for a single .jsonl file before splitting.
                                       Defaults to 200,000,000 bytes.
        """
        self.deployment_name = deployment_name
        self.output_dirpath = output_dirpath
        self.base_filename = base_filename
        self.max_file_size_bytes = max_file_size_bytes
        self.id = uuid4().hex[:5]
        
        self._file_handle: Optional[TextIO] = None
        self._task_id_counter: int = 0
        self.current_file_index: int = 0
        self._current_filepath: Optional[str] = None
        self.generated_files: Set[str] = set()


    def _get_current_output_filename(self) -> str:
        """Gets the filename for the current batch file part."""
        name, ext = os.path.splitext(self.base_filename)
        if not ext: # if base_filename is "requests", default to ".jsonl"
            ext = ".jsonl"
        return f"{self.id}_{name}_{self.current_file_index}{ext}"

    def _ensure_file_open(self):
        """
        Ensures the correct output file (based on current_file_index) is open.
        Updates self._current_filepath and self._file_handle.
        Resets task ID counter if the target file is new or empty.
        Creates the output directory if it doesn't exist.
        """
        target_filename = self._get_current_output_filename()
        target_filepath = os.path.join(self.output_dirpath, target_filename)

        if self._file_handle and not self._file_handle.closed and self._file_handle.name == target_filepath:
            # Already open to the correct file
            if self._current_filepath != target_filepath: # Should ideally not happen if logic is sound
                 self._current_filepath = target_filepath # Ensure consistency
            return

        # If a file is open, but it's not the target (e.g., after index increment), close it
        if self._file_handle and not self._file_handle.closed:
            self.close_file() # This sets self._file_handle to None

        self._current_filepath = target_filepath
        os.makedirs(self.output_dirpath, exist_ok=True)

        is_new_or_empty = not os.path.exists(self._current_filepath) or \
                          os.path.getsize(self._current_filepath) == 0
        
        self._file_handle = open(self._current_filepath, "a", encoding="utf-8")
        self.generated_files.add(self._current_filepath)

        if is_new_or_empty:
            self._task_id_counter = 0
        # else: _task_id_counter continues from current generator instance's state if appending
        # to an existing non-empty file. This is fine for a single generator run creating splits.
        # A new file part in a sequence (e.g., _1.jsonl after _0.jsonl) will be "new or empty"
        # and thus correctly reset the counter.


    def add_request(
        self,
        prompt: str,
        temperature: float = 0.5,
        format: bool = False,
        image_paths: list[str] | None = None,
        custom_id: Optional[str] = None,
    ) -> str:
        """
        Generates a JSONL line for a single request, appends it to the current
        output file, and handles file splitting if necessary.

        Args:
            prompt (str): The text prompt.
            temperature (float): Sampling temperature. Default is 0.5.
            format (bool): If True, requests JSON output format from the model.
            image_paths (list[str] | None): Optional list of paths to local image files.
            custom_id (Optional[str]): A custom identifier for the task. If None,
                                       a unique ID like "task-N" will be generated.

        Returns:
            str: The JSONL string that was written to the file.

        Raises:
            IOError: If the file cannot be written to.
        """
        messages_payload = self._build_messages_payload(prompt=prompt, image_paths=image_paths)

        body: dict[str, Any] = {
            "model": self.deployment_name,
            "messages": messages_payload,
        }

        if self.deployment_name in {"o1", "o3-mini"} or \
           any(model_alias in self.deployment_name.lower() for model_alias in {"o1", "o3-mini"}):
            body["temperature"] = 1.0
        else:
            body["temperature"] = temperature

        if format:
            body["response_format"] = {"type": "json_object"}

        current_custom_id = custom_id if custom_id is not None else f"task-{self._task_id_counter}"

        batch_request_line = {
            "custom_id": current_custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": body,
        }

        json_line_string = json.dumps(batch_request_line)
        encoded_line = json_line_string.encode('utf-8') # For size calculation
        line_size_with_newline = len(encoded_line) + 1 # +1 for newline character

        # 1. Ensure a file is open (or the correct one is open for the current index)
        self._ensure_file_open()
        # self._current_filepath and self._file_handle are now set correctly.

        # 2. Check if writing this line to the *current* file would exceed the max size.
        #    Only split if the file already has content. A single huge line will be written
        #    to an otherwise empty file.
        current_file_size_on_disk = os.path.getsize(self._current_filepath) if self._current_filepath and os.path.exists(self._current_filepath) else 0


        if current_file_size_on_disk > 0 and \
           (current_file_size_on_disk + line_size_with_newline > self.max_file_size_bytes):
            # Current file (already containing data) will become too large.
            # Need to switch to the next file part.
            self.close_file() # Close current file
            self.current_file_index += 1
            self._ensure_file_open() # Open the new file part; this will also reset 
                                     # _task_id_counter if the new file part is indeed new/empty.
        
        if self._file_handle is None or self._file_handle.closed: # Should not be reachable if _ensure_file_open works
             raise IOError("Output file is not open. Call _ensure_file_open() or use as a context manager.")

        self._file_handle.write(json_line_string + "\n")
        self._file_handle.flush()

        if custom_id is None:
            self._task_id_counter += 1

        return json_line_string

    def _build_messages_payload(
        self, prompt: str, image_paths: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        user_message_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt.strip()}
        ]
        if image_paths:
            for image_path in image_paths:
                encoded_image = AzureBatchJsonlGenerator._encode_image(image_path)
                if encoded_image:
                    user_message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    })
        return [{"role": "user", "content": user_message_content}]

    @staticmethod
    def _encode_image(image_path: str) -> Optional[str]:
        max_size_bytes = 20 * 1024 * 1024
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image path does not exist: {image_path}")
                return None
            image_size = os.path.getsize(image_path)

            if image_size > max_size_bytes:
                print(f"Info: Image {image_path} ({image_size / (1024*1024):.2f}MB) exceeds 20MB, attempting resize.")
                with Image.open(image_path) as img:
                    if img.mode == 'RGBA' or img.mode == 'P':
                        img = img.convert('RGB')
                    scale_factor = (max_size_bytes / image_size) ** 0.5
                    new_dimensions = (max(1, int(img.width * scale_factor)), max(1, int(img.height * scale_factor)))
                    img_resized = img.resize(new_dimensions, Image.LANCZOS)
                    
                    buffer = BytesIO()
                    quality = 85
                    img_resized.save(buffer, format="JPEG", quality=quality)
                    buffer_size = buffer.tell()
                    while buffer_size > max_size_bytes and quality > 10:
                        quality -= 10
                        buffer = BytesIO()
                        img_resized.save(buffer, format="JPEG", quality=quality)
                        buffer_size = buffer.tell()
                        print(f"Info: Resized image still too large, reducing quality to {quality}. New size: {buffer_size / (1024*1024):.2f}MB")
                    
                    if buffer_size > max_size_bytes:
                        print(f"Error: Could not resize image {image_path} to be under 20MB even at lowest quality.")
                        return None
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def close_file(self):
        """Closes the current output file if it's open."""
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()
        self._file_handle = None # Mark as closed/requires reopening

    def __enter__(self) -> AzureBatchJsonlGenerator:
        self._ensure_file_open() # Open the initial file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file() # Close the last opened file

    def __repr__(self) -> str:
        return (f"<AzureBatchJsonlGenerator deployment='{self.deployment_name}' "
                f"output_dir='{self.output_dirpath}' base_file='{self.base_filename}' "
                f"current_index='{self.current_file_index}'>")

# --- Example Usage ---
if __name__ == "__main__":
    import shutil

    # --- Test basic functionality and context manager ---
    print("--- Testing with context manager (no splitting expected) ---")
    output_directory = "batch_output_test"
    base_file_name = "requests.jsonl"
    
    # Clean up previous test directory if it exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    deployment_name_for_batch = "gpt-4.1-2"

    with AzureBatchJsonlGenerator(
        deployment_name=deployment_name_for_batch,
        output_dirpath=output_directory,
        base_filename=base_file_name
    ) as batch_gen:
        batch_gen.add_request(prompt="When was Microsoft founded?", temperature=0.7)
        batch_gen.add_request(prompt="Describe this image.", temperature=0.2, custom_id="image_task_alpha")
        batch_gen.add_request(prompt="Generate a JSON object describing a user.", format=True)
    
    print(f"Batch requests generated. Files created: {batch_gen.generated_files}")
    for file_path in batch_gen.generated_files:
        print(f"\nContents of {file_path}:")
        with open(file_path, "r") as f:
            for line in f:
                print(line, end="")
    print("-" * 30)

    # --- Test file splitting ---
    print("\n--- Testing file splitting with small max_file_size_bytes ---")
    split_output_directory = "batch_output_split_test"
    split_base_file_name = "split_requests.jsonl"
    
    if os.path.exists(split_output_directory):
        shutil.rmtree(split_output_directory)
    os.makedirs(split_output_directory, exist_ok=True)

    # Set a very small max file size to force splitting (e.g., 200 bytes)
    # A typical request line might be ~150-300 bytes depending on prompt.
    # This will cause a split after the first or second request.
    small_max_size = 250 

    with AzureBatchJsonlGenerator(
        deployment_name=deployment_name_for_batch,
        output_dirpath=split_output_directory,
        base_filename=split_base_file_name,
        max_file_size_bytes=small_max_size 
    ) as split_gen:
        print(f"Using max_file_size_bytes: {small_max_size}")
        # Request 1 (task-0 in file_0)
        line1 = split_gen.add_request(prompt="Short prompt 1.", temperature=0.1)
        print(f"Added line (len ~{len(line1.encode('utf-8')) + 1}): {line1.strip()}")
        print(f"Current file: {split_gen._current_filepath}, task_id_counter: {split_gen._task_id_counter}")

        # Request 2 (task-1 in file_0, or task-0 in file_1 if split)
        line2 = split_gen.add_request(prompt="This is a slightly longer prompt for request 2.", temperature=0.2)
        print(f"Added line (len ~{len(line2.encode('utf-8')) + 1}): {line2.strip()}")
        print(f"Current file: {split_gen._current_filepath}, task_id_counter: {split_gen._task_id_counter}")
        
        # Request 3 (likely in a new file)
        line3 = split_gen.add_request(prompt="Prompt for request three, this should be another line.", temperature=0.3)
        print(f"Added line (len ~{len(line3.encode('utf-8')) + 1}): {line3.strip()}")
        print(f"Current file: {split_gen._current_filepath}, task_id_counter: {split_gen._task_id_counter}")

        # Request 4 (custom ID, likely in same file as req 3 or new one)
        line4 = split_gen.add_request(prompt="Final prompt 4.", custom_id="my-final-task", temperature=0.4)
        print(f"Added line (len ~{len(line4.encode('utf-8')) + 1}): {line4.strip()}")
        print(f"Current file: {split_gen._current_filepath}, task_id_counter: {split_gen._task_id_counter}")


    print(f"\nBatch requests with splitting generated. Files created: {split_gen.generated_files}")
    for file_path in sorted(list(split_gen.generated_files)): # Sort for consistent output
        print(f"\nContents of {file_path}:")
        file_size = os.path.getsize(file_path)
        print(f"(Size: {file_size} bytes)")
        with open(file_path, "r") as f:
            for line_num, line_content in enumerate(f):
                print(f"  L{line_num+1}: {line_content.strip()}")
                # Verify task IDs are reset for new files
                data = json.loads(line_content)
                if "task-0" in data.get("custom_id","") and line_num > 0 :
                    print(f"    WARNING: custom_id 'task-0' found at line {line_num+1} which is not the first line. Check logic.")
                if "task-0" in data.get("custom_id","") and line_num == 0 :
                    print(f"    INFO: custom_id 'task-0' found at line {line_num+1}. This is expected for a new file part.")
    print("-" * 30)

    # --- Test appending to existing files (if generator is re-used or run again) ---
    # This part is more conceptual as a single script run usually means one "batch generation session".
    # If you ran the script again with the same output_dirpath and base_filename, it would append.
    # The current _task_id_counter logic would restart from 0 for a *new instance* of the generator
    # if it finds an existing, non-empty file, potentially causing custom_id clashes if not careful.
    # However, for file splitting *within one generator instance*, task IDs correctly reset for new file parts.

    # Example: Create a dummy image for testing image encoding
    if not os.path.exists("dummy_image.jpg"):
        try:
            img = Image.new('RGB', (60, 30), color = 'red')
            img.save("dummy_image.jpg")
            print("\nCreated dummy_image.jpg for testing.")
        except Exception as e:
            print(f"Could not create dummy_image.jpg: {e}")
    
    image_test_dir = "batch_output_image_test"
    if os.path.exists(image_test_dir):
        shutil.rmtree(image_test_dir)

    print("\n--- Testing with image ---")
    with AzureBatchJsonlGenerator(
        deployment_name=deployment_name_for_batch,
        output_dirpath=image_test_dir,
        base_filename="image_reqs.jsonl"
    ) as img_gen:
        if os.path.exists("dummy_image.jpg"):
            img_gen.add_request(prompt="What is in this image?", image_paths=["dummy_image.jpg"])
            print(f"Image request added to {img_gen._current_filepath}")
        else:
            print("dummy_image.jpg not found, skipping image test.")

    for file_path in img_gen.generated_files:
        print(f"\nContents of {file_path}:")
        with open(file_path, "r") as f:
            print(f.read().strip()[:300] + "...") # Print beginning of file
    print("-" * 30)

    print("\nCleanup: Removing test directories and dummy image.")
    if os.path.exists(output_directory): shutil.rmtree(output_directory)
    if os.path.exists(split_output_directory): shutil.rmtree(split_output_directory)
    if os.path.exists(image_test_dir): shutil.rmtree(image_test_dir)
    if os.path.exists("dummy_image.jpg"): os.remove("dummy_image.jpg")
    print("Done.")