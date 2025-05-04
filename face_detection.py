import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit the program")
    
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range
        min_detection_confidence=0.5
    ) as face_detection:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect faces
            results = face_detection.process(rgb_frame)
            
            # Draw face detections
            if results.detections:
                for detection in results.detections:
                    # Draw the face detection box
                    mp_drawing.draw_detection(frame, detection)
                    
                    # Get face data
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                int(bbox.width * iw), int(bbox.height * ih)
                    
                    # Display confidence score
                    confidence = detection.score[0]
                    cv2.putText(frame, f'Confidence: {confidence:.2f}',
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Get key points
                    keypoints = detection.location_data.relative_keypoints
                    for keypoint in keypoints:
                        kx, ky = int(keypoint.x * iw), int(keypoint.y * ih)
                        cv2.circle(frame, (kx, ky), 2, (0, 255, 0), -1)
            
            # Display the frame
            cv2.imshow('Face Detection with MediaPipe', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 