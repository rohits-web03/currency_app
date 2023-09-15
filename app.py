import cv2
import streamlit as st
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import numpy as np
import pyttsx3

# Load YOLOv5 model
weights = 'best2.pt'  # Change to the path of your downloaded weights file
device = select_device('')
model = attempt_load(weights, device)
stride = int(model.stride.max())

engine = pyttsx3.init()
# Initialize webcam capture
cap = cv2.VideoCapture(0)


st.title("Currency Recognition with YOLOv5")

# Create a Streamlit canvas for displaying video frames
frame_canvas = st.image([], channels="BGR")

while True:
    curr = 'This note cannot be recognized. Please hold it correctly.'
    ret, frame = cap.read()
    if not ret:
        st.error("Error capturing video from the webcam.")
        break

    # Perform object detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to the model's input size (640x640)
    img = cv2.resize(img, (640, 640))

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # Adjust image dimensions

    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.45, 0.5)

    # Draw bounding boxes on the frame
    for det in pred[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = str(cls)
        if cls == 'tensor(0.)':
            curr = 'This is a Ten Rupees note.'
        elif cls == 'tensor(1.)':
            curr = 'This is a Twenty Rupees note.'
        elif cls == 'tensor(2.)':
            curr = 'This is a Fifty Rupees note.'
        elif cls == 'tensor(3.)':
            curr = 'This is a Hundred Rupees note.'
        elif cls == 'tensor(4.)':
            curr = 'This is a Two Hundred Rupees note.'
        elif cls == 'tensor(5.)':
            curr = 'This is a Five Hundred Rupees note.'
        elif cls == 'tensor(6.)':
            curr = 'This is a Two Thousand Rupees note.'
        st.write(curr)
        engine.say(curr)
        engine.runAndWait()

    # Display the frame with object detection
    frame_canvas.image(frame, channels="BGR")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
