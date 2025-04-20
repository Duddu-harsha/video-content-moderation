import os
import cv2
import numpy as np
import joblib
import argparse
from tqdm import tqdm
import pandas as pd
import glob
# Import the feature extraction function from the main script
# You need to copy the feature extraction functions from your training script
# For simplicity, I'll include the key function below

def extract_hog_features(frame):
    """Extract HOG (Histogram of Oriented Gradients) features from a frame"""
    # Resize frame for consistent feature size
    resized = cv2.resize(frame, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # HOG parameters
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    hog_features = hog.compute(gray)
    return hog_features.flatten()

def extract_orb_features(frame, max_keypoints=100):
    """Extract ORB (Oriented FAST and Rotated BRIEF) features"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    
    # Detect keypoints
    keypoints = orb.detect(gray, None)
    
    # If no keypoints found, return zeros
    if not keypoints:
        return np.zeros(32 * max_keypoints)
    
    # Compute ORB descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)
    
    # If descriptors is None, return zeros
    if descriptors is None:
        return np.zeros(32 * max_keypoints)
    
    # Ensure we have a fixed size by padding or truncating
    if len(keypoints) > max_keypoints:
        descriptors = descriptors[:max_keypoints]
    elif len(keypoints) < max_keypoints:
        padding = np.zeros((max_keypoints - len(keypoints), 32), dtype=np.uint8)
        descriptors = np.vstack((descriptors, padding)) if len(keypoints) > 0 else padding
    
    return descriptors.flatten()

def extract_motion_features(frames):
    """Extract motion-based features between consecutive frames"""
    if len(frames) < 2:
        return np.zeros(4)  # Return zeros if not enough frames
        
    motion_features = []
    for i in range(len(frames) - 1):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude and direction of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Extract statistics from magnitude
        mean_motion = np.mean(magnitude)
        std_motion = np.std(magnitude)
        max_motion = np.max(magnitude)
        motion_direction_entropy = -np.sum(np.histogram(angle, bins=8, density=True)[0] * 
                                          np.log2(np.histogram(angle, bins=8, density=True)[0] + 1e-10))
        
        motion_features.append([mean_motion, std_motion, max_motion, motion_direction_entropy])
    
    # Return mean of motion features across all frame pairs
    return np.mean(np.array(motion_features), axis=0)

def extract_color_statistics(frames):
    """Extract color-based statistics from frames"""
    color_features = []
    for frame in frames:
        # Split into color channels
        b, g, r = cv2.split(frame)
        
        # Calculate statistics for each channel
        for channel in [b, g, r]:
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            skew_val = np.mean(((channel - mean_val) / (std_val + 1e-10)) ** 3)  # Skewness
            entropy = -np.sum(np.histogram(channel, bins=256, density=True)[0] * 
                             np.log2(np.histogram(channel, bins=256, density=True)[0] + 1e-10))
            
            color_features.extend([mean_val, std_val, skew_val, entropy])
        
        # Calculate grayscale histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / np.sum(hist)  # Normalize
        color_features.extend(hist)
    
    # Return mean across all frames
    return np.mean(np.array(color_features).reshape(-1, 28), axis=0)

def extract_edge_features(frame):
    """Extract edge-based features using Canny edge detector"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector with different thresholds
    edges_50_150 = cv2.Canny(blurred, 50, 150)
    edges_100_200 = cv2.Canny(blurred, 100, 200)
    
    # Calculate edge density
    density_50_150 = np.sum(edges_50_150) / (edges_50_150.shape[0] * edges_50_150.shape[1])
    density_100_200 = np.sum(edges_100_200) / (edges_100_200.shape[0] * edges_100_200.shape[1])
    
    # Edge orientation histogram
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi  # Convert to degrees
    
    # Create orientation histogram (8 bins)
    hist, _ = np.histogram(orientation, bins=8, range=(-180, 180), density=True)
    
    # Return combined edge features
    return np.array([density_50_150, density_100_200, *hist])

def extract_video_features(video_path, max_frames=30):
    """Extract comprehensive features from a video without using deep learning"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample frames evenly throughout the video
    frames_to_sample = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        print(f"No frames extracted from: {video_path}")
        return None
    
    # Extract HOG features (from first, middle and last frame)
    hog_frames = [frames[0], frames[len(frames)//2], frames[-1]]
    hog_features = []
    for frame in hog_frames:
        hog_features.append(extract_hog_features(frame))
    hog_features = np.mean(hog_features, axis=0)
    
    # Extract ORB features (from key frames)
    orb_frames = [frames[0], frames[len(frames)//2], frames[-1]]
    orb_features = []
    for frame in orb_frames:
        orb_features.append(extract_orb_features(frame, max_keypoints=20))
    orb_features = np.mean(orb_features, axis=0)
    
    # Extract motion, color and edge features
    motion_features = extract_motion_features(frames)
    color_features = extract_color_statistics(frames)
    
    # Extract edge features from sample frames
    edge_sample_frames = [frames[0], frames[len(frames)//4], frames[len(frames)//2], 
                         frames[3*len(frames)//4], frames[-1]]
    edge_features = []
    for frame in edge_sample_frames:
        edge_features.append(extract_edge_features(frame))
    edge_features = np.mean(edge_features, axis=0)
    
    # Video metadata features
    metadata_features = np.array([
        total_frames,
        fps,
        duration,
        width,
        height,
        width/height  # Aspect ratio
    ])
    
    # Calculate scene change rate
    scene_changes = 0
    if len(frames) > 1:
        prev_hist = cv2.calcHist([cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
        for i in range(1, len(frames)):
            curr_hist = cv2.calcHist([cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
            hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            if hist_diff < 0.8:  # Threshold for scene change
                scene_changes += 1
            prev_hist = curr_hist
        scene_change_rate = scene_changes / (len(frames) - 1)
    else:
        scene_change_rate = 0
    
    # Combine all features
    combined_features = np.concatenate([
        hog_features[:100],           # Limit HOG features to avoid curse of dimensionality
        orb_features[:100],           # Limit ORB features
        motion_features,              # Motion patterns capture activity level
        color_features,               # Color statistics help identify visual tone
        edge_features,                # Edge information
        metadata_features,            # Video metadata
        [scene_change_rate]           # Scene change rate
    ])
    
    return combined_features

def analyze_video(video_path, model_path="video_classifier_v2.pkl"):
    """Analyze a single video using the trained model"""
    # Load model
    model = joblib.load(model_path)
    
    # Extract features
    print(f"Extracting features from {os.path.basename(video_path)}...")
    features = extract_video_features(video_path)
    
    if features is None:
        return {
            "video": os.path.basename(video_path),
            "status": "Error",
            "prediction": None,
            "confidence": None,
            "message": "Failed to extract features"
        }
    
    # Make prediction
    try:
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0, 1]  # Probability of being unsafe
        
        result = {
            "video": os.path.basename(video_path),
            "status": "Success",
            "prediction": "Unsafe" if prediction == 1 else "Safe",
            "confidence": float(probability),
            "message": None
        }
        
        return result
    except Exception as e:
        return {
            "video": os.path.basename(video_path),
            "status": "Error",
            "prediction": None,
            "confidence": None,
            "message": str(e)
        }

def batch_analyze_videos(video_folder, model_path="video_classifier_v2.pkl", output_file="analysis_results.csv"):
    """Analyze all videos in a folder"""
    # Check if folder exists
    if not os.path.exists(video_folder):
        print(f"Error: Folder {video_folder} does not exist")
        return
    
    # List video files
    video_files = []
    for ext in [".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"]:
        video_files.extend(glob.glob(os.path.join(video_folder, f"*{ext}")))
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    print(f"Found {len(video_files)} video files to analyze")
    
    # Analyze each video
    results = []
    for video_path in tqdm(video_files, desc="Analyzing videos"):
        result = analyze_video(video_path, model_path)
        results.append(result)
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Analysis complete. Results saved to {output_file}")
    
    # Print summary
    if len(results) > 0:
        safe_count = sum(1 for r in results if r["prediction"] == "Safe" and r["status"] == "Success")
        unsafe_count = sum(1 for r in results if r["prediction"] == "Unsafe" and r["status"] == "Success")
        error_count = sum(1 for r in results if r["status"] == "Error")
        
        print(f"\nSummary:")
        print(f"  Safe videos: {safe_count}")
        print(f"  Unsafe videos: {unsafe_count}")
        print(f"  Processing errors: {error_count}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Content Analysis Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", "-v", help="Path to a single video file to analyze")
    group.add_argument("--folder", "-f", help="Path to a folder of videos to analyze")
    parser.add_argument("--model", "-m", default="video_classifier_v2.pkl", help="Path to the trained model")
    parser.add_argument("--output", "-o", default="analysis_results.csv", help="Output CSV file for batch analysis")
    
    args = parser.parse_args()
    
    # Process based on arguments
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file {args.video} does not exist")
            return
        
        result = analyze_video(args.video, args.model)
        if result["status"] == "Success":
            print(f"\nAnalysis Result for {os.path.basename(args.video)}:")
            print(f"  Classification: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            
            # Add more detailed interpretation
            if result["prediction"] == "Unsafe":
                risk_level = "High" if result["confidence"] > 0.8 else "Moderate" if result["confidence"] > 0.6 else "Low"
                print(f"  Risk Level: {risk_level}")
                print(f"  Recommendation: {'Immediate review recommended' if risk_level == 'High' else 'Review recommended'}")
            else:
                print(f"  Risk Level: {'Very Low' if result['confidence'] < 0.2 else 'Low'}")
                print("  Recommendation: Content appears safe")
        else:
            print(f"Error analyzing video: {result['message']}")
    
    elif args.folder:
        import glob  # Import here to avoid unused import if not needed
        batch_analyze_videos(args.folder, args.model, args.output)

if __name__ == "__main__":
    main()