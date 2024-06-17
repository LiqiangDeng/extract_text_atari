import os
import sys
import pytesseract
import pandas as pd
import numpy as np
import cv2
from download_video import download_video_name


# Path to pytesseract executable (change this according to your system)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to extract text from image using pytesseract
def extract_text(image, roi=None, debug=False):
    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
    
    # image = cv2.resize(image,(0,0),fx=2,fy=7)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        show_image('Gray Image', gray)

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    if debug:
        show_image('Blur Image', blur)
        
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if debug:
        show_image('Threshold Image', thresh)

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug:
        show_image('Opening Image', opening)

    invert = 255 - opening
    if debug:
        show_image('Inverted Image', invert)

    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    # config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(invert, config=config)
    return text


if __name__ == "__main__":

    xlsx_path = 'data/game_dataset.xlsx'
    df = pd.read_excel(xlsx_path)
    filtered_df = df.dropna(subset=['bounding_box'])[['name', 'yt_link', 'bounding_box']]
    print(filtered_df)

    for index, row in filtered_df.iterrows():
        if index != 1:
            break
        name = row['name']
        yt_link = row['yt_link']
        bounding_box = eval(row['bounding_box'])

        video_dir = os.path.join("data", name)
        os.makedirs(video_dir, exist_ok=True)
        
        # check and download video
        video_filename = f"{name}.mp4"
        video_path = os.path.join(video_dir, f'{name}.mp4')
        if not os.path.exists(video_path):
            print(f"File not found: {video_filename}")
            video_url = yt_link

            download_video_name(video_url, video_filename, video_dir)

        # start to read video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        mappings = []

        skip_seconds = 6 * 60
        cap.set(cv2.CAP_PROP_POS_MSEC, skip_seconds * 1000)

        # Loop through frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('not ret')
                break

            roi = bounding_box
            x, y, w, h = roi
            bottom_right_x = x + w
            bottom_right_y = y + h

            text = extract_text(frame, roi, debug=True)

            # Print the extracted text
            print(text)

            # save frame
            frame_filename = f"frame_{frame_count:04d}.png"
            frame_path = os.path.join(video_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            # save mapping
            mappings.append(f"{frame_filename}, {text}")

            # draw red box in frame
            cv2.rectangle(frame, (x, y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)

            # Display the frame (optional)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        # Release video capture object
        print('release')
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # save mapping
        mappings_path = os.path.join(video_dir, "mappings.txt")
        with open(mappings_path, "w") as f:
            f.write("\n".join(mappings))

        print(f"Processed {frame_count} frames for video {video_filename}")