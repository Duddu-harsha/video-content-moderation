import os

# Folder where videos are saved
video_folder = r"C:\Users\hp\Desktop\videosdata\videos"

# Total expected videos
total_videos = 2000

# Generate expected filenames: Video_0001.mp4 to Video_2000.mp4
expected_videos = [f"Video_{i:04d}.mp4" for i in range(1, total_videos + 1)]

# Actual video files present in the folder
present_videos = set(os.listdir(video_folder))

# Find missing videos
missing_videos = [video for video in expected_videos if video not in present_videos]

# Output
print("Missing Videos:")
for mv in missing_videos:
    print(mv)

# Optional: Save to a text file
with open("missing_videos.txt", "w") as f:
    for mv in missing_videos:
        f.write(mv + "\n")

print(f"\nTotal missing: {len(missing_videos)}")

