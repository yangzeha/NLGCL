import os
import requests
import zipfile
import shutil

def download_and_extract_dataset(dataset_name, save_dir='dataset'):
    # RecBole uses S3 accelerate URLs
    if dataset_name.lower() == 'yelp':
        url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp.zip"
    elif dataset_name.lower() == 'yelp2018':
        url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp2018.zip"
    elif dataset_name.lower() == 'pinterest':
        url = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Pinterest/pinterest.zip"
    else:
        print(f"Unknown dataset: {dataset_name}")
        return

    target_dir = os.path.join(save_dir, dataset_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check for .inter files (RecBole atomic files)
    has_inter = False
    for f in os.listdir(target_dir):
        if f.endswith('.inter'):
            has_inter = True
            break
            
    if has_inter:
        print(f"Dataset {dataset_name} seems to be already present in {target_dir}. Skipping.")
        return

    return_path = os.getcwd()
    
    zip_path = os.path.join(target_dir, f"{dataset_name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Downloading {dataset_name} from {url}...")
    try:
        # Check if partial file exists for resuming
        resume_header = {}
        mode = 'wb'
        if os.path.exists(zip_path):
            existing_size = os.path.getsize(zip_path)
            # Fetch header to get total size first
            head_resp = requests.head(url)
            total_size_server = int(head_resp.headers.get('content-length', 0))
            
            if existing_size < total_size_server:
                print(f"Resuming download from {existing_size} bytes...")
                resume_header = {'Range': f'bytes={existing_size}-'}
                mode = 'ab'
                downloaded = existing_size
            elif existing_size == total_size_server:
                print("Download already complete. Proceeding to extraction.")
                # Configure fake response object for logic flow downstream if needed, 
                # or just skip download loop.
                # Simplified: Just set flag to skip download
                downloaded = existing_size
                total_size = total_size_server
                response = None 
            else:
                # Local file larger? Delete and restart.
                os.remove(zip_path)
                downloaded = 0
        else:
            downloaded = 0

        if 'response' not in locals() or response is None:
             if mode == 'ab' or mode == 'wb':
                response = requests.get(url, headers=resume_header, stream=True)
                # Note: 206 Partial Content is success for Range request
                if mode == 'ab' and response.status_code != 206:
                     # Server doesn't support range, restart
                     print("Server doesn't support resume. Restarting download.")
                     mode = 'wb'
                     downloaded = 0
                else:
                    response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0)) + downloaded if response else total_size_server
        
        block_size = 8192
        
        if response:
            with open(zip_path, mode) as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) < block_size:
                            print(f"Downloaded {downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB", end='\r')

        print(f"\nDownload complete. Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)
            print(f"Extracted to {target_dir}")
        os.remove(zip_path)
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        # Do not remove zip_path on error to allow resume
        print("Download interrupted. File kept for resuming.")

if __name__ == "__main__":
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    download_and_extract_dataset('Yelp')
