import openai
import cv2
import base64
from pytubefix import YouTube
from dotenv import load_dotenv
import os
import pandas as pd
from pytubefix.cli import on_progress
from pytube import cipher
import re

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',
    ]
    #logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            #logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(
        caller="get_throttling_function_name", pattern="multiple"
    )

def download_video(video_url, output_path=None, video_name=None):
    try:
        yt = YouTube(video_url, on_progress_callback = on_progress)
        stream = yt.streams.get_highest_resolution()

        if output_path is None:
            output_path = os.getcwd()
            
        if video_name is None:
            video_name = stream.default_filename

        stream.download(output_path=output_path, filename=video_name)

        print(f"Download completed: {video_name}")
    except Exception as e:
        print(f"An error occurred while downloading {video_url}: {str(e)}")

def extract_frame_from_video(video_path, output_dir="frames", frame_rate=5, start_time=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if start_time:
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting from frame {start_frame} (time: {start_time} seconds).")
    else:
        start_frame = 0

    frame_count = start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps // frame_rate) == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame to {frame_path}")
            break 
        frame_count += 1

    cap.release()
    return frame_path if 'frame_path' in locals() else None

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

def detect_bounding_boxes_with_chatgpt(image_path, prompt):
    image_base64 = encode_image_to_base64(image_path)

    messages = [
        {"role": "system", "content": "You are an object detection assistant. You analyze Atari video game frames and return bounding boxes for objects."},
        {"role": "user", "content": f"This is a Base64 encoded image:\n{image_base64}\n\n{prompt}\nPlease return the bounding box results in JSON format using (x, y, w, h)."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response['choices'][0]['message']['content']
    print(reply)
    return reply

def process_csv(input_csv, output_csv, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    frame_folder = os.path.join(output_folder, "frames")
    os.makedirs(frame_folder, exist_ok=True)

    df = pd.read_csv(input_csv)

    for index, row in df.iterrows():
        video_url = row['yt_link']
        bounding_box = row['bounding box (x, y, w, h)'] if 'bounding box (x, y, w, h)' in row else None
        duration = row['duration']
        video_name = "temp_video.mp4"

        if pd.isna(video_url) or not video_url.startswith("https"):
            print(f"Skipping invalid URL at index {index}")
            continue

        if not pd.isna(bounding_box) and bounding_box.strip() != "":
            print(f"Skipping row {index} as bounding box already exists.")
            continue

        try:
            duration = float(duration)
        except ValueError:
            print(f"Invalid duration for video at index {index}. Defaulting to 0.")
            duration = 0

        middle_time = duration / 2 if not pd.isna(duration) else 0
        print(f"Processing video at middle time {middle_time} seconds.")

        video_path = os.path.join(output_folder, "temp_video.mp4")

        try:
            download_video(video_url, output_path=output_folder, video_name=video_name)

            frame_path = extract_frame_from_video(video_path, output_dir="frames", frame_rate=5, start_time=middle_time)

            if frame_path:
                prompt = """
                This is a frame from an Atari video game. In this frame, there is a reward object visible. 
                The reward object is typically the area most relevant to the player receiving a reward. 

                Please detect and return the most likely reward object's bounding box in the format (x, y, w, h). 
                Only return one bounding box, which is the best match for the reward. 
                If there is no clear reward object, return an empty JSON object: {}.
                """

                bounding_box_result = detect_bounding_boxes_with_chatgpt(frame_path, prompt)
                
                if not bounding_box_result or not bounding_box_result.strip():
                    print(f"No bounding box result for frame at index {index}. Storing empty result...")
                    df.at[index, 'bounding box (raw result)'] = "{}"
                else:
                    print(f"Storing bounding box result for frame at index {index}.")
                    df.at[index, 'bounding box (raw result)'] = bounding_box_result
        except Exception as e:
            print(f"Error processing video {video_url}: {e}")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(frame_folder):
                for frame_file in os.listdir(frame_folder):
                    os.remove(os.path.join(frame_folder, frame_file))
            print(f"Cleaned up temporary files for video at index {index}.")

    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "ll_dataset - atari365.csv"
    output_csv = "atari_365_output_chatgpt.csv"
    output_folder = 'tmp'

    process_csv(input_csv, output_csv, output_folder)