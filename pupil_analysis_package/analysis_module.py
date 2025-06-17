import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_video(video_path, output_dir):
    """Processes one video and saves the results"""
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_dir, f"{video_name}_results.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open the video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Couldn't get FPS for video {video_path}")
        cap.release()
        return None

    data = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Processing \'{video_name}\'") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                      param1=50, param2=30, minRadius=10, maxRadius=30)

            left_diameter = None
            right_diameter = None

            if circles is not None:
                circles = np.uint16(np.around(circles[0]))

                if len(circles) == 2:
                    circles = sorted(circles, key=lambda x: x[0])
                    left_diameter = 2 * circles[0][2]
                    right_diameter = 2 * circles[1][2]
                else:
                    left_diameter = 2 * circles[0][2]

            time = frame_count / fps
            data.append({
                'Time': time,
                'Diameter of the left pupil (px)': left_diameter,
                'Diameter of the right pupil (px)': right_diameter
            })

            frame_count += 1
            pbar.update(1)

    cap.release()

    df = pd.DataFrame(data)
    df = df.dropna(subset=['Diameter of the left pupil (px)', 'Diameter of the right pupil (px)'], how='any')
    df.to_csv(output_csv, index=False)
    print(f"The results for {video_name} are saved in {output_csv}")

    return df

def process_videos_in_folder_recursive(input_root, output_root):
    """Recursively processes the video and saves the CSV in a similar structure."""
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.MOV')

    results = {}

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_root)

                output_dir = os.path.join(output_root, relative_path)

                df = process_video(input_path, output_dir)

                if df is not None:
                    results[input_path] = df

    return results


