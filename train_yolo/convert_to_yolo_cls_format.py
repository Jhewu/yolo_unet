import os
import csv

def create_labels_csv(directory_path):
    """
    Create a labels.csv file from a directory structure.
    
    Args:
        directory_path (str): Path to the directory containing subdirectories with images
    """
    # Path to the output labels.csv file
    output_file = os.path.join(directory_path, 'labels.csv')
    
    # List to store all rows for the CSV
    csv_rows = []
    
    # Iterate through subdirectories (0, 1, 2, ...)
    for subdir_name in sorted(os.listdir(directory_path)):
        subdir_path = os.path.join(directory_path, subdir_name)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            try:
                # Convert subdir name to integer label
                label = int(subdir_name)
                
                # Get all image files in the subdirectory
                for filename in sorted(os.listdir(subdir_path)):
                    # Check if it's an image file (you can add more extensions if needed)
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        csv_rows.append([filename, str(label)])
                        
            except ValueError:
                # Skip directories that aren't numeric
                continue
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    
    print(f"Created labels.csv with {len(csv_rows)} entries")

# Example usage:
create_labels_csv('verifier_dataset/test')