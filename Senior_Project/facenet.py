import cv2
import numpy as np
import os
from PIL import Image
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data
import tensorflow as tf

# Initialize the face detection model
detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)

# Load the TFLite model from the file
tflite_model = open("facenet.tflite", "rb")
interpreter = tf.lite.Interpreter(model_content=tflite_model.read())

# Allocate the tensors
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to preprocess face images for Facenet model
def extract_embeddings(image_np):

    # Preprocess the image
    image_np = tf.image.resize(image_np, (160, 160))
    image_np = image_np / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    # Set the data for the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image_np)

    # Run the interpreter.
    interpreter.invoke()

    # Get the output
    output = interpreter.get_tensor(output_details[0]['index'])

    return output


# Load images of known persons and their names
known_faces = {}
for file_name in os.listdir("kids2"):
    img_path = os.path.join("kids2", file_name)
    img = cv2.imread(img_path)
    known_faces[file_name.split('.')[0]] = extract_embeddings(img)

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to the appropriate index if you have multiple cameras

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    # Convert the OpenCV frame to a PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect faces in the frame
    faces = detect_faces(frame_pil)

    # Check if any faces are detected
    if not len(faces):
        print('No faces detected :(')
    else:
        # Convert face detections to render data
        annotations = detections_to_render_data(faces, bounds_color=Colors.GREEN)

        for annotation in annotations:
            closest_distance = float('inf')
            closest_label = None
            closes_face_path = ""
            for item in annotation.data:
                left, right, top, bottom = (int(item.left * frame.shape[1]), int(item.right * frame.shape[1]),
                                            int(item.top * frame.shape[0]), int(item.bottom * frame.shape[0]))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                face_img = frame[top:bottom, left:right]

                try:

                    # Extract embeddings for the detected face
                    embeddings = extract_embeddings(face_img)

                except:
                    continue

                # Compare embeddings with known faces
                for Id, known_embedding in known_faces.items():
                    distance = np.linalg.norm(embeddings - known_embedding)  # getting eucludian distance from
                    if distance < closest_distance:
                        closest_distance = distance  # reassign to closest distance
                        closest_label = Id
                        closes_face_path = "kids2/" + closest_label + ".jpg"

            # Display the label of the closest match
            if closest_label is not None:
                cv2.putText(frame, closest_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f"Similarity: {1 - (closest_distance / 100):.2f}", (left, top - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imshow("similar face", cv2.imread(closes_face_path))

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

