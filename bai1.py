import cv2
from urllib.request import urlopen
import numpy as np
url =r'http://172.16.70.160/capture'

while True:
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    img = cv2.imdecode(imgnp, -1)
    # image = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == 113:
        break

# Destroy all the windows
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
from urllib.request import urlopen
import numpy as np

model = YOLO("yolo11n.pt")
url =r'http://172.16.70.160/capture'
while True:
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    image = cv2.imdecode(imgnp, -1)
    # image = cv2.rotate(img, cv2.ROTATE_180)
    # Run inference with the YOLO11n model on the 'bus.jpg' image
    results = model(image)
    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
            confidence = box.conf[0].item() # Confidence score
            class_id = int(box.cls[0]) # Class ID
            if class_id == 0:
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'person {confidence:.2f}', (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display the image
    cv2.imshow("YOLO Detection", image) # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()