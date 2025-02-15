# This file is used to manually check whether the bounding boxes in images and videos are accurate

import cv2

image_path = 'Bakery Simulator - Ultimate Workshop Episode 5/frame_000480.png'
image = cv2.imread(image_path)

bounding_box = (1450, 40, 220, 60)

x, y, w, h = bounding_box

# draw red rect
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Image with Bounding Box", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2

# video_path = 'videos/Cosmic_Commuter.mp4'

# # skip time in seconds
# skip_time = 40

# bounding_box = (120, 54, 100, 20)
# x, y, w, h = bounding_box

# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Cannot open video.")
#     exit()

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# skip_frames = int(skip_time * fps)

# cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or cannot read frame.")
#         break

#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.imshow("Video with Bounding Box", frame)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()