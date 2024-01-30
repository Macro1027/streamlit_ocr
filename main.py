import cv2
import time
import queue
import threading
import streamlit as st

from scripts.preprocess import preprocess_img
from scripts.ocr import ocr_thread
from scripts.chatbot import run_chatbot

def initialize_queues():
    # Create and return all necessary queues for the application.
    return {
        'frame_queue': queue.Queue(maxsize=1),
        'text_queue': queue.Queue(maxsize=1),
        'prompt_queue': queue.Queue(maxsize=1),
        'ppx_queue': queue.Queue(maxsize=1)
    }

def start_threads(queues):
    # Start OCR and chat completion threads.
    threading.Thread(target=ocr_thread, args=(queues['frame_queue'], queues['text_queue'])).start()

def capture_video():
    # Set up and return video capture object.
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return video_capture

def display_ocr_results(frame, text_queue, conf_thresh=50, text_placeholder=None):
    # Display OCR results on the frame if available.
    if not text_queue.empty():

        d = text_queue.get()

        n_boxes = len(d)
        # Loop through each text box
        to_append = []
        for i in range(n_boxes):
            for j in range(len(d[i]['text'])):
                detected_text = d[i]['text'][j]

                # Check confidence level and profanity of detected text
                if int(d[i]['conf'][j]) > conf_thresh:
                    to_append.append(detected_text)
                    (x, y, w, h) = (d[i]['left'][j], d[i]['top'][j], d[i]['width'][j], d[i]['height'][j])
                    if x and y and w and h:

                        # Display bounding box and text
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, detected_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if len(to_append) > 0:
            text_placeholder.text(to_append)

def main():
    # Initialise queues and start threads
    queues = initialize_queues()
    start_threads(queues)

    # Create streamlit widgets
    st.title('Webcam Live Feed')
    FRAME_WINDOW = st.image([])
    run = st.checkbox('Run')
    text_placeholder = st.empty()
    conf_thresh = st.slider('Confidence Threshold', 0, 100, 50)
    run_chatbot()

    # Start video capture
    cap = capture_video()
    while run:
        ret, frame = cap.read()

        start = time.perf_counter()

        if not ret:
            break

        frame = cv2.resize(frame, (840, 480))

        # Display OCR results on the frame if available.
        display_ocr_results(frame, queues['text_queue'], conf_thresh, text_placeholder)

        # Add frame to queue if it is empty
        if queues['frame_queue'].empty():
            preprocessed_img = preprocess_img(frame)
            queues['frame_queue'].put(preprocessed_img)


        end = time.perf_counter()
        fps = 1/(end - start)

        # Display frame with FPS
        cv2.putText(frame, f'FPS: {fps:.2f}', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        FRAME_WINDOW.image(frame, channels="BGR")
    else:
        st.write('Stopped')
        
    # Release reesources
    cap.release()
    cv2.destroyAllWindows()


# Plan - 
#    1. Make the text stay longer
#    2. Allow questions to be asked to ppx

if __name__ == '__main__':
    main()
