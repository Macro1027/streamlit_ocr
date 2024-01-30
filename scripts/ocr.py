from pytesseract import image_to_data, Output, image_to_string

import sys
import os

# Add the directory containing this script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Perform OCR in a separate thread
def ocr_thread(frame_queue, text_queue):
    from frozen_east_detection import detect_boxes

    while True:
        texts = []
        frame = frame_queue.get()

        # If queue is empty, exit the loop
        if frame is None:
            break
 
        # Retrieve text bounding boxes from EAST
        boxes = detect_boxes(frame)

        for (startX, startY, endX, endY) in boxes:

            try:
                cropped = frame[startY:endY, startX:endX]
                text = image_to_data(cropped, output_type=Output.DICT)
                for i in range(len(text['text'])):
                    text['left'][i] += startX
                    text['top'][i] += startY

            except Exception as e:
                continue

            texts.append(text)
            text_queue.put(texts)

if __name__ == "__main__":
    import sys
    print(sys.path)
