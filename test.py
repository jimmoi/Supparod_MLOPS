import os
import shutil

# --- Configuration ---
SOURCE_ROOT_DIR = r'Dataset\test' # The directory containing class_1, class_2, etc.
DESTINATION_ROOT_DIR = r'Dataset_github' # The new directory to copy files into
FILES_TO_COPY = 10
FILE_EXTENSION = '.jpg' # The extension of the files you want to copy

# Create the destination root directory if it doesn't exist
os.makedirs(DESTINATION_ROOT_DIR, exist_ok=True)
print(f"Destination root directory created: {DESTINATION_ROOT_DIR}")

# Iterate through all entries in the source root directory
for class_name in os.listdir(SOURCE_ROOT_DIR):
    # Construct the full path to the current class directory
    source_class_dir = os.path.join(SOURCE_ROOT_DIR, class_name)

    # Check if the entry is actually a directory (a class folder)
    if os.path.isdir(source_class_dir):
        # 1. Create the corresponding destination class directory
        destination_class_dir = os.path.join(DESTINATION_ROOT_DIR, class_name)
        os.makedirs(destination_class_dir, exist_ok=True)
        
        print(f"\nProcessing class: {class_name}")

        # 2. Get all image files in the current class directory
        # The listdir output is generally not sorted, so we sort it alphabetically
        # to ensure "first 10" is consistently determined (e.g., by filename).
        all_files = sorted(os.listdir(source_class_dir))
        
        # Filter for the desired image extension
        image_files = [
            f for f in all_files 
            if f.lower().endswith(FILE_EXTENSION) and os.path.isfile(os.path.join(source_class_dir, f))
        ]
        
        # 3. Select the first N files
        files_to_copy = image_files[:FILES_TO_COPY]

        # 4. Copy the selected files
        files_copied_count = 0
        for filename in files_to_copy:
            source_file_path = os.path.join(source_class_dir, filename)
            destination_file_path = os.path.join(destination_class_dir, filename)
            
            # Use shutil.copy2 to copy the file and its metadata
            shutil.copy2(source_file_path, destination_file_path)
            files_copied_count += 1
            # Optional: print for verification
            # print(f"  Copied: {filename}")
            
        print(f"Successfully copied {files_copied_count} images to {destination_class_dir}")

print("\nProcess complete.")