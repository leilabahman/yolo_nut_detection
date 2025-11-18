from ultralytics import YOLO
import cv2
import time

# Load trained model
model = YOLO("../results/nut_detection/weights/best.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference
    results = model.predict(frame, verbose=False)[0]

    results = model(frame, conf=0.6)
    annotated = results[0].plot()
    
    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLO Nut Detection", annotated)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
