import cv2
import numpy as np


def get_score_region_by_color_sege(frame):
    """
    Predict the coordinates of the score display area based on color segmentation and shape features.

    param:
    frame (numpy.array): Video frames, BGR format.

    return:
    tuple: The coordinates of the fractional region (x, y, w, h), if not found, are None.

    """
    # Convert images to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for the color of the score area (assuming yellow here)
    lower_color = np.array([20, 100, 100])
    upper_color = np.array([30, 255, 255])
    
    # color segmentation
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Applying morphological operations (dilation and closure) to remove noise and connect characters
    # kernel_close = np.ones((5, 15), np.uint8)  # Wide and flat rectangular structural elements
    # morph_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Additional dilation operation to further connect characters
    # kernel_dilate = np.ones((5, 15), np.uint8)  # Smaller wide and flat rectangular structural elements
    # morph_dilate = cv2.dilate(morph_close, kernel_dilate, iterations=2)

    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    
    # Find the contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    possible_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        print(x, y, w, h)
        
        # Assuming the score area is not particularly small
        if w > 10 and h > 10:
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + 2 * margin
            h = h + 2 * margin
            possible_regions.append((x, y, w, h))
    
    # If multiple candidate regions are found, choose the one with the largest area
    if possible_regions:
        possible_regions.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
        return possible_regions[0]
    
    return None

def get_score_region_by_image_process(frame):
    """
    Predicting the coordinates of the score display area based on image processing technology

    param:
    frame (numpy.array): Video frames, BGR format.

    return:
    tuple: The coordinates of the fractional region (x, y, w, h), if not found, are None.

    """
    # Convert image to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Application binarization threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Morphological operations (expansion and corrosion) to remove noise
    kernel = np.ones((10, 10), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Display processed images
    cv2.imshow('Morph', morph)
    cv2.waitKey(0)
    
    # Find the contour
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    possible_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Assuming the score area is not particularly small
        if w > 15 and h > 10:
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + 2 * margin
            h = h + 2 * margin
            possible_regions.append((x, y, w, h))
    
    # If multiple candidate regions are found, choose the one with the largest area
    if possible_regions:
        possible_regions.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
        return possible_regions[0]
    
    return None