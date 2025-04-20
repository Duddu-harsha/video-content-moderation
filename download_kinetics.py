import os
import cv2
import pandas as pd
from yt_dlp import YoutubeDL

# File paths
csv_path = r"C:\Users\hp\Desktop\videosdata\val_sampled_5_per_category.csv"
download_dir = r"C:\Users\hp\Desktop\videosdata\temp_downloads"
final_dir = r"C:\Users\hp\Desktop\videosdata\videos"

# Create necessary folders
os.makedirs(download_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

# yt-dlp options
ydl_opts = {
    'format': 'mp4',
    'outtmpl': os.path.join(download_dir, '%(id)s.%(ext)s'),
    'quiet': True,
    'noplaylist': True
}

print("🚀 Starting download and trimming from CSV...")

try:
    # Load CSV
    df = pd.read_csv(csv_path)

    # Loop through rows with a counter for Video_0001 naming
    for index, row in df.iterrows():
        video_number = f"Video_{index+1:04d}"
        youtube_id = str(row['youtube_id']).strip()
        start_sec = int(row['time_start'])
        end_sec = int(row['time_end'])

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        temp_path = os.path.join(download_dir, f"{youtube_id}.mp4")
        final_path = os.path.join(final_dir, f"{video_number}.mp4")

        print(f"\n🎬 Processing {video_number}: {youtube_id} | {start_sec}s to {end_sec}s")

        try:
            # Download video
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Open video
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(final_path, fourcc, fps, (width, height))

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame >= start_frame and current_frame < end_frame:
                    out.write(frame)
                elif current_frame >= end_frame:
                    break

                current_frame += 1

            cap.release()
            out.release()
            print(f"✅ Saved as: {final_path}")

            # Delete the temp file to save space
            os.remove(temp_path)

        except Exception as e:
            print(f"❌ Failed {video_number}: {youtube_id} | Error: {e}")

except KeyboardInterrupt:
    print("\n⛔ Stopped by user.")
except Exception as e:
    print(f"\n⚠️ Unexpected Error: {e}")

print("\n🎉 All videos processed!")
