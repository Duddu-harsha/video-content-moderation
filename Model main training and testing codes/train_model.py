import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === CONFIGURATION ===
videos_path = r"C:\Users\hp\Desktop\videosdata\videos"
csv_path = r"C:\Users\hp\Desktop\contentmodcode\val.csv"
model_save_path = "video_classifier_v2.pkl"
features_cache_dir = "feature_cache"  # Directory to cache extracted features
max_frames = 30  # Sample frames evenly throughout video
n_jobs = 4        # Number of parallel processes for feature extraction

# Create cache directory if it doesn't exist
os.makedirs(features_cache_dir, exist_ok=True)

# === FEATURE EXTRACTION FUNCTIONS ===
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

def extract_video_features(video_path, cache_path=None):
    """Extract comprehensive features from a video without using deep learning"""
    # Check if features are already cached
    if cache_path and os.path.exists(cache_path):
        return joblib.load(cache_path)
    
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
    
    # Cache the features
    if cache_path:
        joblib.dump(combined_features, cache_path)
    
    return combined_features

def process_video(args):
    """Process a single video (for parallel processing)"""
    video_id, video_path = args
    cache_path = os.path.join(features_cache_dir, f"{video_id}_features.pkl")
    
    try:
        features = extract_video_features(video_path, cache_path)
        return video_id, features
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        return video_id, None

# === VIDEO FINDER FUNCTION ===
def find_video_file(video_id, videos_dir):
    for ext in [".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"]:
        full_path = os.path.join(videos_dir, video_id + ext)
        if os.path.exists(full_path):
            return full_path
    return None

# === MAIN PROCESSING PIPELINE ===
def main():
    start_time = time.time()
    print("🚀 Starting CPU-friendly video content classification pipeline...")
    
    # Load data
    print("📦 Reading CSV data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return
    
    # Validate and clean data
    print("🧹 Validating and cleaning data...")
    df = df.dropna(subset=['Video_id', 'Label'])
    
    # Convert labels to standard format
    df['Label'] = df['Label'].str.strip().str.lower()
    valid_labels = ['safe', 'unsafe']
    df = df[df['Label'].isin(valid_labels)]
    
    print(f"✅ After cleaning: {len(df)} valid records")
    print(f"📊 Class distribution: {df['Label'].value_counts().to_dict()}")
    
    # Extract features from videos in parallel
    print("🎬 Extracting features from videos (this may take a while)...")
    video_tasks = []
    
    for idx, row in df.iterrows():
        video_id = row['Video_id']
        video_file = find_video_file(video_id, videos_path)
        
        if video_file:
            video_tasks.append((video_id, video_file))
        else:
            print(f"❌ Video not found: {video_id}")
    
    # Process videos in parallel
    X = {}
    y = {}
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_video, task) for task in video_tasks]
        
        with tqdm(total=len(futures), desc="Processing Videos") as pbar:
            for future in as_completed(futures):
                video_id, features = future.result()
                if features is not None:
                    label_idx = df[df['Video_id'] == video_id].index[0]
                    label = df.loc[label_idx, 'Label']
                    
                    X[video_id] = features
                    y[video_id] = 1 if label == 'unsafe' else 0
                pbar.update(1)
    
    # Convert to lists while maintaining alignment
    video_ids = list(X.keys())
    X_list = [X[vid] for vid in video_ids]
    y_list = [y[vid] for vid in video_ids]
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n📊 Total videos successfully processed: {len(X)} / {len(df)}")
    
    # Check if we have enough data
    if len(X) < 10:
        print("❗ Not enough videos processed to train a reliable model!")
        return
    
    # Train the model with proper preprocessing and hyperparameter tuning
    print("\n🔍 Training and optimizing machine learning model...")
    
    # Create train/test split with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"]))
    
    print("\n📊 Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"\n📊 AUC-ROC Score: {auc_roc:.4f}")
    
    # Save the model
    print(f"\n💾 Saving trained model to: {model_save_path}")
    joblib.dump(best_model, model_save_path)
    
    # Calculate feature importance
    if hasattr(best_model[-1], 'feature_importances_'):
        importances = best_model[-1].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n🔍 Top 10 most important features:")
        for i in range(min(10, len(importances))):
            print(f"  Feature #{indices[i]}: {importances[indices[i]]:.4f}")
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print("\n✅ Done! The enhanced video content classification system is ready.")
    
    # Save video IDs and their predicted probabilities for review
    results_df = pd.DataFrame({
        'video_id': [video_ids[i] for i in range(len(X_test))],
        'actual': ['Unsafe' if y == 1 else 'Safe' for y in y_test],
        'predicted': ['Unsafe' if y == 1 else 'Safe' for y in y_pred],
        'unsafe_probability': y_prob
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("\n💾 Prediction results saved to: prediction_results.csv")
    
    # Provide example code for using the model
    print("\n📝 Example usage for inference:")
    print("""
    # Load the model
    import joblib
    import cv2
    import numpy as np
    model = joblib.load("video_classifier_v2.pkl")
    
    # Extract features from a new video
    from your_module import extract_video_features  # Import your feature extraction function
    video_path = "path/to/video.mp4"
    features = extract_video_features(video_path)
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0, 1]
    
    print(f"Prediction: {'Unsafe' if prediction == 1 else 'Safe'}")
    print(f"Confidence: {probability:.2f}")
    """)

if __name__ == "__main__":
    main()