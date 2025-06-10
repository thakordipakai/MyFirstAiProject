import openai
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# Set up OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key

# Function to detect faces using OpenCV
def detect_face(image_path):
    # Load the OpenCV pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If a face is detected, return the coordinates of the first face
    if len(faces) > 0:
        return faces[0]  # (x, y, w, h) of the first detected face
    return None

# Function to describe image using OpenAI API
def describe_image(image_path):
    # If face detected, proceed
    face = detect_face(image_path)
    if face is not None:
        # For now, let's send the whole image for description
        image = Image.open(image_path)
        image.show()  # Open the image for inspection

        # Open image and convert to base64 (or can directly send as binary)
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()

        # Use OpenAI API to describe the image
        response = openai.Image.create(
            file=img_data,
            purpose="description"
        )
        
        # Extract the description from OpenAI's response
        description = response['data'][0]['text']

        print(f"Description: {description}")
    else:
        print("No face detected.")

# Main function to run the detection and description
def main(image_path):
    describe_image(image_path)

# Example usage
image_path = "your_image_path.jpg"  # Replace with your image path
main(image_path)
