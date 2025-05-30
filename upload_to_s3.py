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


def upload_folder_to_s3(local_folder_path, bucket_name, s3_prefix=""):
    """
    Uploads an entire folder (including subdirectories and files) to an S3 bucket.

    :param local_folder_path: Path to the local folder to upload.
    :param bucket_name: Name of the S3 bucket.
    :param s3_prefix: Optional S3 object key prefix (folder name in S3).
                      If provided, files will be uploaded under this prefix.
                      e.g., "my_uploads/".
    :return: True if the folder was uploaded successfully, else False.
    """
    if not os.path.isdir(local_folder_path):
        print(f"Error: Local folder not found or is not a directory at '{local_folder_path}'. Cannot upload.")
        return False

    print(f"Attempting to upload folder '{local_folder_path}' to S3 bucket '{bucket_name}' under prefix '{s3_prefix}'...")

    s3_client = boto3.client('s3')
    upload_success = True

    # Walk through the local directory
    for root, dirs, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            
            # Construct the S3 object key.
            # This preserves the relative path from the original local_folder_path.
            # For example, if local_folder_path is 'data' and file is 'data/sub/file.txt',
            # relative_path will be 'sub/file.txt'.
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            
            # Combine the S3 prefix with the relative path to form the full S3 object key.
            # Ensure consistent path separators for S3 (forward slashes).
            s3_object_key = os.path.join(s3_prefix, relative_path).replace(os.sep, '/')

            try:
                print(f"Uploading '{local_file_path}' to s3://{bucket_name}/{s3_object_key}...")
                s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
                print(f"Successfully uploaded '{local_file_path}'")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                error_message = e.response.get("Error", {}).get("Message")
                
                if error_code == 'NoSuchBucket':
                    print(f"Error: The S3 bucket '{bucket_name}' does not exist or you don't have access to it.")
                elif error_code == 'AccessDenied':
                    print(f"Error: Access denied. Your AWS credentials do not have permission to upload to '{bucket_name}'.")
                else:
                    print(f"Error uploading file '{local_file_path}' to S3: {e}")
                    print(f"AWS Error Code: {error_code}, Message: {error_message}")
                upload_success = False
                # Continue trying other files even if one fails
            except Exception as e:
                print(f"An unexpected error occurred while uploading '{local_file_path}': {e}")
                upload_success = False
                # Continue trying other files even if one fails

    if upload_success:
        print(f"\nFolder '{local_folder_path}' upload process completed. All files successfully uploaded.")
    else:
        print(f"\nFolder '{local_folder_path}' upload process completed with some errors.")

    return upload_success

if __name__ == "__main__":
    # Call the upload function with your specified folder, bucket, and prefix
    LOCAL_FOLDER_TO_UPLOAD = ""
    upload_folder_to_s3(LOCAL_FOLDER_TO_UPLOAD, AWS_BUCKET_NAME, "")


if __name__ == "__main__":
    # Ensure the JSON file path is correct
    json_path_to_upload = DEFAULT_JSON_FILENAME

    # Call the upload function
    upload_json_to_s3(json_path_to_upload, AWS_BUCKET_NAME, S3_OBJECT_KEY)