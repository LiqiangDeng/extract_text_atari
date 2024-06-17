import cv2
import pytesseract
import sys
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from get_region_contour import get_score_region_by_color_sege, get_score_region_by_image_process

# Path to pytesseract executable (change this according to your system)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'


# Function to extract text from image using pytesseract
def extract_text(image, roi=None):
    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
        column_sums = np.sum(image, axis=0)
    else:
        column_sums = np.sum(image, axis=0)


    threshold = 800
    non_empty_columns = np.where(column_sums > threshold)[0]

    # Determine the left and right boundaries of ROI
    if non_empty_columns.size:
        left_index = non_empty_columns[3]
        right_index = non_empty_columns[-2]
        left_bound = left_index - 3
        right_bound = right_index + 3
        # Adjust the upper and lower boundaries y and h of ROI as needed
        y = 0
        h = image.shape[0]  # Assuming the entire image height
        # Update ROI
        roi = (left_bound, y, right_bound-left_bound, h)
    else:
        roi = None  # No non blank columns found, special handling may be required

    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
    
    image = cv2.resize(image,(0,0),fx=2,fy=7)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening 

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
        # roi = get_score_region_by_color_sege(frame)
        # if not roi:
        #     continue
        # print(roi)

        # Atari 2600 Longplay Demon Attack boundary determined by manual
        # roi = (200, 25, 135, 20)

        roi = (5, 7, 70, 15)
        x, y, w, h = roi
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
# video_path = 'Atari 2600 Longplay [066] Demon Attack.mp4'
process_video(video_path)