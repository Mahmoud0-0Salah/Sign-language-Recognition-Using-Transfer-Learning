import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import torchvision.models as models
import preprocess

labels = ['Fine', 'Hello', 'I love you', 'Water', 'You']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = preprocess.preprocess_img(image, transform)
        # Predict the label
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            print(outputs, predicted, predicted[0])
            predicted_label = labels[predicted.item()]
            print("Predicted label:", predicted_label)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
