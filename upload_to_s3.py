import os
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
#DEFAULT_JSON_FILENAME = "/mnt/data_volume/dataset/downloaded_valid_pdfs/collected_pdf_texts.json"
DEFAULT_JSON_FILENAME = "data/collected_pdf_texts.json"

AWS_BUCKET_NAME = "digitizepid"
S3_OBJECT_KEY = "STF_con_sum.jsonl" # The name it will have in the S3 bucket

def upload_json_to_s3(json_file_path, bucket_name, object_key):
    """
    Uploads a file to an S3 bucket.
    :param json_file_path: Path to the file to upload.
    :param bucket_name: Name of the S3 bucket.
    :param object_key: S3 object key (filename in S3).
    :return: True if file was uploaded, else False.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'. Cannot upload.")
        return False

    print(f"Attempting to upload '{json_file_path}' to S3 bucket '{bucket_name}' as '{object_key}'...")

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(json_file_path, bucket_name, object_key)
        print(f"Successfully uploaded '{json_file_path}' to s3://{bucket_name}/{object_key}")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        
        if error_code == 'NoSuchBucket':
            print(f"Error: The S3 bucket '{bucket_name}' does not exist or you don't have access to it.")
        elif error_code == 'AccessDenied':
            print(f"Error: Access denied. Your AWS credentials do not have permission to upload to '{bucket_name}'.")
        else:
            print(f"Error uploading file to S3: {e}")
            print(f"AWS Error Code: {error_code}, Message: {error_message}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Ensure the JSON file path is correct
    json_path_to_upload = DEFAULT_JSON_FILENAME

    # Call the upload function
    upload_json_to_s3(json_path_to_upload, AWS_BUCKET_NAME, S3_OBJECT_KEY)