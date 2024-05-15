import os
import preprocess


# Define input and output directories
input_dir = 'data'
output_dir = 'hand_data'

# Iterate over subfolders in the input directory
for subdir, _, files in os.walk(input_dir):
    # Create corresponding subfolder in the output directory
    output_subdir = os.path.join(output_dir, os.path.relpath(subdir, input_dir))
    os.makedirs(output_subdir, exist_ok=True)
    
    # Iterate over files in the subfolder
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            # Extract hand region and save it
            input_image_path = os.path.join(subdir, file)
            output_image_path = os.path.join(output_subdir, file)
            hand_region = preprocess.extract_hand_region(input_image_path)
            if hand_region is not None:
                hand_region.save(output_image_path)
