import os
import time
from google.cloud import storage

def file_exists_in_gcs(bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    return blob.exists()


def upload_to_gcs(bucket_name, destination_blob_name, source_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded: {source_file_name} -> gs://{bucket_name}/{destination_blob_name}")

def monitor_and_upload(local_dir, bucket_name, interval=3600):
    
    for filename in os.listdir(local_dir):
        file_path = os.path.join(local_dir, filename)
        destination_blob_name = f"{filename}"

        if not os.path.isfile(file_path):
            continue
        
        try:
            if file_exists_in_gcs(bucket_name, destination_blob_name):
                print(f"Skipped: {filename} already exists in GCS.")
                continue
            
            upload_to_gcs(bucket_name, destination_blob_name, file_path)
            
            
        except Exception as e:
            print(f"Error uploading {filename}: {e}")
    

if __name__ == "__main__":
    local_directory = "/root/diffusion-learning/weights_super_resolution"  # Directory where model weights are saved
    bucket_name = "private-diffusion-weights"

    os.makedirs(local_directory, exist_ok=True)

    print(f"Monitoring {local_directory} for new files...")
    monitor_and_upload(local_directory, bucket_name)
