import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# Define known living things
living_things = ['person', 'dog', 'cat', 'bird', 'cow', 'horse', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe']

# Define a function to classify object
def is_living(label):
    return "Living Thing" if label.lower() in living_things else "Non-Living Thing"

# Sample object info database
object_info = {
    "laptop": {"cost": "₹40,000 - ₹1,00,000", "size": "13-17 inches"},
    "bottle": {"cost": "₹20 - ₹300", "size": "0.5 - 2L"},
    "person": {"cost": "N/A", "size": "5 - 6 ft"},
    "chair": {"cost": "₹500 - ₹5000", "size": "3 - 4 ft"},
    "dog": {"cost": "₹2000 - ₹20,000", "size": "1 - 2.5 ft"},
    "cell phone": {"cost": "₹5,000 - ₹1,00,000", "size": "5 - 7 inches"},
    "book": {"cost": "₹100 - ₹1000", "size": "8 x 11 inches"},
    # Add more objects as needed
}

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(frame, verbose=False)[0]

    # Process detections
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = map(int, result)
        label = model.names[cls]

        # Classify as Living or Non-Living
        status = is_living(label)

        # Fetch object info
        info = object_info.get(label.lower(), {"cost": "Unknown", "size": "Unknown"})

        # Draw rectangle around object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw object name
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw AR info
        ar_text = f"{status} | Cost: {info['cost']} | Size: {info['size']}"
        cv2.putText(frame, ar_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show result
    cv2.imshow("VisionAR - Intelligent Object Scanner", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()