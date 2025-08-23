import re
import os
import requests
import pandas as pd

# Path to the input text file
input_file = "Links_Updated.txt"
# Directory to save downloaded files
download_folder = "downloaded_camera_files_New"
# Output CSV file
output_csv = "camera_file_ids.csv"

# Create the download folder if it doesn't exist
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Regular expression to match Camera1 and Camera2 links
pattern = r'<a href="(https://drive\.google\.com/a/up\.edu\.mx/uc\?id=[^&]+&amp;export=download)">Camera[12]</a>'

# Read the input file
try:
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"Error: Input file not found: {input_file}")
    exit(1)
except Exception as e:
    print(f"Error reading input file {input_file}: {e}")
    exit(1)

# Find all matching links
links = re.findall(pattern, content)
if not links:
    print(f"No matching links found in {input_file}")
    exit(1)

# List to store link and filename data
file_data = []

# Function to download a file from a Google Drive link and collect metadata
def download_file(url, folder):
    # Extract file ID from the URL
    match = re.search(r'id=([^&]+)', url)
    if not match:
        print(f"Invalid URL format, no file ID found: {url}")
        return None, None
    
    file_id = match.group(1)
    # Construct the direct download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Send a HEAD request to get the file name from headers
    try:
        response = requests.head(download_url, allow_redirects=True, timeout=10)
    except requests.RequestException as e:
        print(f"Error fetching headers for {url}: {e}")
        return url, None
    
    # Extract file name from Content-Disposition header
    file_name = None
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        match = re.search(r'filename="([^"]+)"', content_disposition)
        if match:
            file_name = match.group(1)
    
    # Fallback to a generic name if no filename is found
    if not file_name:
        file_name = f"file_{file_id}.zip"  # Assuming zip, adjust if needed
    
    file_path = os.path.join(folder, file_name)
    
    # Store link and filename
    file_data.append({'Link': url, 'Filename': file_name})
    
    # Check if file already exists, skip to avoid overwriting
    if os.path.exists(file_path):
        print(f"File already exists, skipping: {file_path}")
        return url, file_name
    
    # Download the file
    try:
        response = requests.get(download_url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
            return url, file_name
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return url, file_name
    
    return url, file_name

# Download each file and collect metadata
for link in links:
    download_file(link, download_folder)

# Create a DataFrame and save to CSV
if file_data:
    df = pd.DataFrame(file_data, columns=['Link', 'Filename'])
    try:
        df.to_csv(output_csv, index=False)
        print(f"CSV file created: {output_csv} with {len(df)} entries")
    except Exception as e:
        print(f"Error writing CSV file {output_csv}: {e}")
else:
    print("No file data to write to CSV")

print(f"Total files processed: {len(links)}")
print(links)