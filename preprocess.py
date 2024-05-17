import cv2
from PIL import Image
import torchvision.transforms.functional as F
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)


# Function to get hand bounding box
def get_hand_bbox(hand_landmarks, image_shape):
    x_coords = [landmark.x * image_shape[1] for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * image_shape[0] for landmark in hand_landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return int(x_min), int(y_min), int(x_max), int(y_max)


# Function to extract hand region and resize
def extract_hand_region(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB format (required by MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return extract_hand_from_img(image_rgb)


def extract_hand_from_img(image):
    # Perform hand detection
    results = hands.process(image)

    # Extract bounding boxes and confidence scores
    if results.multi_hand_landmarks:
        boxes = []
        confidences = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand bounding box
            x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, image.shape)
            boxes.append([x_min, y_min, x_max, y_max])
            confidences.append(results.multi_handedness[0].classification[0].score)

        # Sort bounding boxes based on confidence scores (in descending order)
        sorted_indices = sorted(
            range(len(confidences)), key=lambda i: confidences[i], reverse=True
        )
        sorted_boxes = [boxes[i] for i in sorted_indices]

        hand_box = sorted_boxes[0]
        margin = 20
        expanded_hand_box = [
            max(hand_box[0] - margin, 0),  # Left
            max(hand_box[1] - margin, 0),  # Top
            min(hand_box[2] + margin, image.shape[1]),  # Right
            min(hand_box[3] + margin, image.shape[0]),  # Bottom
        ]
        print(hand_box)
        # Crop the hand region from the image
        image = Image.fromarray(image)
        cropped_image = F.crop(
            image,
            expanded_hand_box[1],
            expanded_hand_box[0],
            expanded_hand_box[3] - expanded_hand_box[1],
            expanded_hand_box[2] - expanded_hand_box[0],
        )
        return cropped_image
    else:
        print("No hands detected in the image")
        return None


def preprocess_img(image, transform):
    hand_extracted = True
    hand = extract_hand_from_img(image)
    if not hand:
        hand = Image.fromarray(image)
        hand_extracted = False

    image = transform(hand)
    image = image.unsqueeze(0)
    return image, hand_extracted
