from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import os

model = YOLO('cube_data/output2/weights/best.pt')

image_path = 'cube_data/testing/images'

images = os.listdir( image_path )

for img_name in images:
    results = model(os.path.join( image_path, img_name ) )

    img = cv2.imread(os.path.join( image_path, img_name ) )
    
    for det in results[0].boxes:
        box = det.xyxy[0].cpu().numpy()  # Bounding box coordinates
        cls = int(det.cls[0])  # Class index
        conf = det.conf[0]  # Confidence score
        label = f"{model.names[cls]} {conf:.2f}"
    
        # Draw the bounding box and label on the image
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("test", img )
    cv2.waitKey(0)
