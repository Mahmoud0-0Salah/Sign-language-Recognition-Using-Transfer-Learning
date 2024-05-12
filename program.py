import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import torchvision.models as models

labels = ['Fine', 'Hello', 'I love you', 'Water', 'You']

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Load the model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load('model_params.pth'))
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)

    # Press 'o' to capture frame and predict
    if cv2.waitKey(1) & 0xFF == ord('o'):
        # Preprocess the captured frame
        preprocessed_frame = preprocess_image(frame)

        # Predict the label
        with torch.no_grad():
            outputs = model(preprocessed_frame)
            _, predicted = torch.max(outputs, 1)
            print(outputs, predicted, predicted[0])
            predicted_label = labels[predicted.item()]
            print("Predicted label:", predicted_label)
            break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
