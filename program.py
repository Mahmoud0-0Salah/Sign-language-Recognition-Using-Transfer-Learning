import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import torchvision.models as models
import preprocess

labels = ['Hello', 'I love you', 'Water', 'You']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
# model = models.resnet18()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4)
# model.load_state_dict(torch.load('model_params.pth'))

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, 4)
model.load_state_dict(torch.load('model_params_mobilenetv2.pth'))
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the captured frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image, hand_extracted = preprocess.preprocess_img(image, transform)
    
    if hand_extracted:
        # Predict the label
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = labels[predicted.item()]

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
