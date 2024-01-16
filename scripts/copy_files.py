import boto3
import os
import threading


def download_file(s3_client, bucket_name, s3_file, local_file):
    """
    Download a single file from S3 to the local file system.

    :param s3_client: Boto3 S3 client.
    :param bucket_name: Name of the S3 bucket.
    :param s3_file: S3 file key.
    :param local_file: Local file path to save the file.
    """
    s3_client.download_file(bucket_name, s3_file, local_file)
    print(f"Downloaded {s3_file} to {local_file}")

def download_directory_from_s3(bucket_name, s3_folder, local_dir, max_threads=10):
    """
    Download the contents of a folder directory from S3 to a local directory.

    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Path to the folder inside the S3 bucket.
    :param local_dir: Local directory to download files to.
    :param max_threads: Maximum number of threads to use for parallel downloads.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    # Create a pool of threads
    threads = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for item in page.get('Contents', []):
            file_key = item['Key']
            if file_key.endswith('/'):
                continue  # Skip folders

            local_file_path = os.path.join(local_dir, file_key[len(s3_folder):].lstrip('/'))

            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Start a thread to download this file
            thread = threading.Thread(target=download_file, args=(s3, bucket_name, file_key, local_file_path))
            thread.start()
            threads.append(thread)

            # Wait if we have too many active threads
            while len(threads) >= max_threads:
                for t in threads:
                    if not t.is_alive():
                        threads.remove(t)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == '__main__':


    # Usage
    bucket_name = 'db-brian-mosaic'

    # annotations_folder
    annotations = 'openmmlab/data/coco/annotations'
    annotations_local_dir = '/tmp/data/annotations'

    train = 'openmmlab/data/coco/train2017'
    train_local_dir = '/tmp/data/train2017'

    val = 'openmmlab/data/coco/val2017'
    val_local_dir = '/tmp/data/val2017'

    test = 'openmmlab/data/coco/test2017'
    test_local_dir = '/tmp/data/test2017'

    download_directory_from_s3(bucket_name, annotations, annotations_local_dir, threads=100)
    download_directory_from_s3(bucket_name, train, train_local_dir, threads=100)
    download_directory_from_s3(bucket_name, val, val_local_dir, threads=100)
    download_directory_from_s3(bucket_name, test, test_local_dir, threads=100)