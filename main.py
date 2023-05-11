import cv2
from paddleocr import PaddleOCR
import numpy as np
import re
import torch

car_model = torch.hub.load('ultralytics/yolov5', 'custom', path='cardetect.pt', force_reload=True)
plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path='main.pt', force_reload=True)

video_path = 'asd.mp4'
cap = cv2.VideoCapture(video_path)

number_plate_pattern = '^[a-zA-Z0-9_.]$'

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # Use the YOLOv5 model to detect cars
    car_results = car_model(frame)

    # Get the bounding boxes and confidence scores for the detected cars
    car_boxes = car_results.xyxy[0][:, :4].cpu().numpy()
    car_scores = car_results.xyxy[0][:, 4].cpu().numpy()

    # Filter out low-confidence car detections
    car_threshold = 0.5
    cars = car_boxes[car_scores >= car_threshold]

    # Loop through the detected cars
    for car in cars:
        # Crop the car region
        x1, y1, x2, y2 = car.astype(np.int32)
        car_image = frame[y1:y2, x1:x2]

        # Use the YOLOv5 model to detect number plates in the car region
        plate_results = plate_model(car_image)

        # Get the bounding boxes and confidence scores for the detected number plates
        plate_boxes = plate_results.xyxy[0][:, :4].numpy()
        plate_scores = plate_results.xyxy[0][:, 4].numpy()

        # Filter out low-confidence number plate detections
        plate_threshold = 0.5
        plates = plate_boxes[plate_scores >= plate_threshold]

        # Loop through the detected number plates
        for plate in plates:
            # Extract the number plate region
            plate_x1, plate_y1, plate_x2, plate_y2 = plate.astype(np.int32)
            number_plate_image = car_image[plate_y1:plate_y2, plate_x1:plate_x2]

            # Apply OCR to recognize the characters on the number plate
            result = ocr_engine.ocr(number_plate_image)
            number_plate_text = ''
            for line in result:
                for word in line:
                    number_plate_text += " "

            number_plate_text = re.sub('^[a-zA-Z0-9_.]$', '', number_plate_text)

            # Check if the recognized text matches the format of a valid number plate
            # Check if the recognized text matches the format of a valid number plate
            valid_plate_format = re.compile('^[a-zA-Z0-9_.]$')
            print('Detected Number Plate:', number_plate_text)
            # Draw bounding box around the detected number plate
            cv2.rectangle(car_image, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 0), 2)
            # Display the detected number plate text on the frame
            cv2.putText(frame, number_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw bounding boxes around the detected cars
            for car in cars:
                x1, y1, x2, y2 = car.astype(np.int32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Display the output image
        cv2.imshow('Number Plate Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
