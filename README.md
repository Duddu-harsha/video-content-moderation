# video-content-moderation
AI model for automated video content moderation with trimming
🚀 How to Use
1. Train the Model
python train_model.py
Reads video metadata and labels from val.csv

Extracts features

Trains a RandomForestClassifier with hyperparameter tuning

Saves model as video_classifier_v2.pkl

2. Analyze a Single Video

python test_video_classifier.py --video path/to/video.mp4 --model video_classifier_v2.pkl
3. Batch Analyze a Folder

python test_video_classifier.py --folder path/to/folder --model video_classifier_v2.pkl --output results.csv
🧠 Feature Extraction
The system extracts a variety of features from videos:

HOG (Histogram of Oriented Gradients)

ORB (Keypoints & Descriptors)

Motion (Optical Flow)

Color statistics (Mean, Std, Skewness, Entropy)

Edge features (Canny edge & Sobel histogram)

Scene change rate

Metadata (fps, resolution, duration, etc.)

📊 Dataset
Stored in val.csv

Expected columns:

Video_id – ID (filename without extension)

Label – Safe or Unsafe

Category – (used in categorynumcalc.py and videoselectioncode.py)

📌 Utilities
categorynumcalc.py: Counts unique categories in the dataset

get_duration.py: Computes durations of videos in a folder

videoselectioncode.py: Picks 5 videos per category for sampling

✅ Example Inference Code
python
Copy
Edit
from test_video_classifier import extract_video_features
import joblib

model = joblib.load("video_classifier_v2.pkl")
features = extract_video_features("your_video.mp4")
prediction = model.predict([features])[0]
confidence = model.predict_proba([features])[0, 1]

print(f"Prediction: {'Unsafe' if prediction == 1 else 'Safe'}")
print(f"Confidence: {confidence:.2f}")
📌 Requirements
Python 3.8+

OpenCV

NumPy

Pandas

Scikit-learn

Joblib

Tqdm
