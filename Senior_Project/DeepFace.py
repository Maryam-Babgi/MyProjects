import cv2
from deepface import DeepFace

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to the appropriate index if you have multiple cameras

while True:

    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    dfs = DeepFace.find(frame,
                        db_path="Kids Faces",
                        model_name="Facenet512",  # For Face Recognition
                        detector_backend="opencv",  # For Face Detection
                        enforce_detection=False,
                        distance_metric="cosine")

    for face in dfs:
        if face.shape[0] > 0:
            x, y, w, h = face.source_x[0], face.source_y[0], face.source_w[0], face.source_h[0]
            if x != 0 and y != 0:
                distance = face.distance[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                most_similar_face = face.iloc[0].identity
                similar_face = cv2.imread(most_similar_face)
                cv2.imshow("similar face", similar_face)
                label = most_similar_face.split("\\")[-1][:-4]
                cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
