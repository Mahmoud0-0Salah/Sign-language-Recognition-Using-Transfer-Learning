import os
import time
import cv2

IMAGES_PATH = 'data/'

labels = ['You', 'Fine', 'Water', 'Hello', 'I love you']
number_imgs = 10

for label in labels:
    # os.makedirs(IMAGES_PATH + label)
    cap = cv2.VideoCapture(0)
    print(f"Collecting images for {label}")
    time.sleep(5)
    for img_num in range(number_imgs):
        ret, frame = cap.read()
        img_path = os.path.join(IMAGES_PATH, label, f"{label}_{img_num}.jpg")
        cv2.imwrite(img_path, frame)
        cv2.imshow('frame', frame)
        time.sleep(4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
