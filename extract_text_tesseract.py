import cv2
import pytesseract
import sys
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Path to pytesseract executable (change this according to your system)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

# Function to extract text from image using pytesseract
def extract_text(image, roi=None):
    # if roi:
    #     x, y, w, h = roi
    #     image = image[y:y+h, x:x+w]

    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
        column_sums = np.sum(image, axis=0)
    else:
        column_sums = np.sum(image, axis=0)

    threshold = 800
    # print(image.shape, column_sums)
    non_empty_columns = np.where(column_sums > threshold)[0]
    # print(non_empty_columns)

    # Determine the left and right boundaries of ROI
    if non_empty_columns.size:
        left_index = non_empty_columns[3]
        right_index = non_empty_columns[-2]
        left_bound = left_index - 3
        right_bound = right_index + 3
        y = 0
        h = image.shape[0]
        roi = (left_bound, y, right_bound-left_bound, h)
    else:
        roi = None

    # print(image.shape, roi)
    # cv2.imshow('Grayscale Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL) 
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # if roi:
    #     x, y, w, h = roi
    #     right_bound = x + w
    #     print(image[y:y+h, x:x+w])
    #     column_sums = np.sum(image[y:y+h, x:right_bound], axis=0)
    #     print(column_sums)
    #     try:
    #         non_zero_columns = np.where(column_sums > 0)[0]
    #         left_bound = non_zero_columns[0] if non_zero_columns.size else right_bound - w
    #     except IndexError:
    #         left_bound = right_bound - w
        
    #     print(left_bound, right_bound)
    #     image = image[y:y+h, left_bound:right_bound]
    
    image = cv2.resize(image,(0,0),fx=2,fy=7)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening 
    # Use pytesseract to do OCR on the grayscale image
    # text = pytesseract.image_to_string(gray)

    # cv2.imshow('Grayscale Image', invert)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    # config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(invert, config=config)
    return text

# Function to process video frames
def process_video(video_path):

    # Open the video file
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
    cap = cv2.VideoCapture(video_path)

    #skip_seconds = 10 
    skip_seconds = 6 * 60
    cap.set(cv2.CAP_PROP_POS_MSEC, skip_seconds * 1000)

    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('not ret')
            break
        # Extract text from the frame
        roi = (200, 25, 135, 20)
        x, y, w, h = (200, 25, 135, 20)
        bottom_right_x = x + w
        bottom_right_y = y + h
        # draw red box in frame
        cv2.rectangle(frame, (x, y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
        text = extract_text(frame, roi)
        # Print the extracted text
        print(text)
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release video capture object
    print('release')
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Path to the video file
video_path = sys.argv[1]

# Process the video
video_path = 'Atari 2600 Longplay [066] Demon Attack.mp4'
process_video(video_path)