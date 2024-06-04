from pytube import YouTube
import sys


def download_video(video_url, output_path=None):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest resolution stream
        stream = yt.streams.get_highest_resolution()

        # If output path is not provided, use default filename
        if output_path is None:
            output_path = stream.default_filename

        # Download the video
        stream.download(output_path=output_path)

        print("Download completed!")
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage
if __name__ == "__main__":
    video_url = sys.argv[1]
    output_path = sys.argv[2]

    download_video(video_url, output_path)