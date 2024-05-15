import os
import cv2

IMAGES_PATH = 'data/'

labels = ['you', 'fine', 'water', 'hello', 'i love you']
number_imgs = 10

for label in labels:
   
    folder_path = os.path.join(IMAGES_PATH, label)
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Consider only image files
            # Read the image
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Create darker version
            darker_img = cv2.subtract(img, 50)  # Adjust this value for desired darkness
            darker_output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_darker.jpg")
            cv2.imwrite(darker_output_path, darker_img)

            # Create lighter version
            lighter_img = cv2.add(img, 50)  # Adjust this value for desired lightness
            lighter_output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_lighter.jpg")
            cv2.imwrite(lighter_output_path, lighter_img)

            print(f"Processed: {filename}")
    print(f"{label} done")

print("All images resized successfully!")
