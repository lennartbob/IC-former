import os
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
AWS_BUCKET_NAME = "digitizepid"
S3_OBJECT_KEY = "SFT_con_sum.jsonl" # The name it will have in the S3 bucket
DEFAULT_JSON_FILENAME = "SFT_con_sum.jsonl" # Local filename for upload and download

def download_json_from_s3(bucket_name, object_key, local_file_path):
    """
    Downloads a file from an S3 bucket and saves it locally.
    :param bucket_name: Name of the S3 bucket.
    :param object_key: S3 object key (filename in S3).
    :param local_file_path: Path where the downloaded file will be saved.
    :return: True if file was downloaded, else False.
    """
    print(f"Attempting to download '{object_key}' from S3 bucket '{bucket_name}' to '{local_file_path}'...")

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Ensure the local directory exists
    local_dir = os.path.dirname(local_file_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Created local directory: '{local_dir}'")

    try:
        s3_client.download_file(bucket_name, object_key, local_file_path)
        print(f"Successfully downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")

        if error_code == 'NoSuchKey':
            print(f"Error: The S3 object '{object_key}' does not exist in bucket '{bucket_name}'.")
        elif error_code == 'NoSuchBucket':
            print(f"Error: The S3 bucket '{bucket_name}' does not exist or you don't have access to it.")
        elif error_code == 'AccessDenied':
            print(f"Error: Access denied. Your AWS credentials do not have permission to download from '{bucket_name}'.")
        else:
            print(f"Error downloading file from S3: {e}")
            print(f"AWS Error Code: {error_code}, Message: {error_message}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return False

if __name__ == "__main__":
    # --- Example Usage ---

    # Define paths
    json_path_to_upload = DEFAULT_JSON_FILENAME
    download_directory = "data"
    local_download_path = os.path.join(download_directory, S3_OBJECT_KEY)
    # --- Download the file ---
    print("\n--- Initiating Download ---")
    download_success = download_json_from_s3(AWS_BUCKET_NAME, S3_OBJECT_KEY, local_download_path)
    if download_success:
        print(f"Download process completed successfully. Check '{local_download_path}'")
    else:
        print("Download process failed.")
