import os
import re

# Path to your folder
folder_path = r"C:\Users\sheew\Downloads\tumors2\Testing\no_tumor"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
        # Extract the number (last 4 digits) using regex
        match = re.search(r"\((\d+)\)\.jpg$", filename)
        if match:
            number = match.group(1)
            new_name = f"no_tumor({number}).jpg"
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

print("✅ Renaming complete!")