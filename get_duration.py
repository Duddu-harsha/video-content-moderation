import cv2
import os

def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open: {video_path}")
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return round(duration, 2)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0.0

def get_all_video_durations(folder_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    durations = {}

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        return durations

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(video_extensions):
            video_path = os.path.join(folder_path, file_name)
            duration = get_video_duration(video_path)
            durations[file_name] = duration

    return durations

# Provide your folder path here
folder_path = r"C:\Users\hp\Desktop\violence dataset\violence"
durations = get_all_video_durations(folder_path)

# Print results
print("\nVideo Durations:")
for video, duration in durations.items():
    print(f"{video} - {duration} seconds")
