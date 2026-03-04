from google.cloud import storage
from datetime import timedelta
import os

# Set your credentials and bucket info
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/yash/Desktop/magiclips deploy/new-backend-475414-228aab3a465a.json"
BUCKET_NAME = "subtitlegen-input"
BLOB_NAME = "5c6dde19-448f-48a3-99e9-6c0f6db5982d.mp4"

def generate_signed_url(bucket_name, blob_name, expiration_minutes=15, method="GET"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method=method,
        content_type="video/mp4" if method == "PUT" else None
    )
    return url

if __name__ == "__main__":
    url = generate_signed_url(BUCKET_NAME, BLOB_NAME)
    print("Signed URL:", url)