import numpy as np
import cv2

def preprocess_img(img):
    # Convert image to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply otsu thresholding - depends on its neighbours
    method, thresh = determine_best_thresholding_method(gray)

    
    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    result = 255 - opening
    result = cv2.GaussianBlur(result, (3,3), 0)

    # Convert image to gbr
    image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return image

def calculate_histogram_bimodality(hist):
    # Calculate the bimodality coefficient of the histogram
    # This is a simplified example and may not be robust in all cases
    peaks = (hist > np.roll(hist, 1)) & (hist > np.roll(hist, -1))
    num_peaks = np.sum(peaks)
    return num_peaks == 2  # Assuming bimodal if exactly two peaks

def determine_best_thresholding_method(img):

    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Check if the histogram is bimodal
    if calculate_histogram_bimodality(hist):
        # If the histogram is bimodal, Otsu's method might be a good choice
        _, thresholded_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method = 'Otsu'
    else:
        # If the histogram is not bimodal, adaptive thresholding might work better
        thresholded_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        method = 'Adaptive'

    return method, thresholded_img

if __name__ == "__main__":
    img = cv2.imread('test5.jpg')

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    preprocessed = preprocess_img(img)
    cv2.imshow('new img', preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
