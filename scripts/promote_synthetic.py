import argparse
from google.cloud import storage

def promote_files(bucket_name, prefix="staging/"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    
    print(f"Scanning {prefix} in {bucket_name}...")
    
    count = 0
    for blob in blobs:
        if blob.name.endswith('.npy'):
            # In a real app, this would present a plot to a user
            # Here we simulate "Approval"
            print(f"Reviewing {blob.name}...")
            
            # Simulate Cardiologist Decision
            # approved = input("Approve? (y/n): ")
            approved = 'y' # Auto-approve for demo
            
            if approved == 'y':
                new_name = blob.name.replace('staging/', 'approved/')
                bucket.copy_blob(blob, bucket, new_name)
                blob.delete()
                print(f"  -> Promoted to {new_name}")
                count += 1
            else:
                print("  -> Rejected")
                blob.delete()
                
    print(f"Promoted {count} synthetic samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    args = parser.parse_args()
    promote_files(args.bucket)
