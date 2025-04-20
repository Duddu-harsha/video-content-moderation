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

                  WRITE UP 


🎯 AI-Powered Automated Video Content Moderation System
🔍 Project Overview
In the age of digital media, moderating video content at scale has become one of the most pressing challenges. Our project aims to tackle this by developing a fully automated, intelligent video moderation system capable of detecting and classifying harmful content with minimal human intervention.

Rather than analyzing videos as single units, our system dissects each video frame-by-frame—typically at a rate of one frame per second. This granular analysis enables us to detect inappropriate or harmful segments even if they occur briefly within longer videos. Each selected frame is analyzed using a combination of handcrafted visual features and machine learning, producing high-confidence predictions that categorize content by type of harm (e.g., violence, explicit material, abuse) and risk level (low, medium, high).

This frame-level approach enables us to deliver detailed, risk-aware moderation that goes beyond simple binary classification. The system integrates smoothly with existing workflows using simple CSV-based mappings and supports both single and batch video analysis without requiring complex infrastructure.

🧠 Technical Approach
Our system follows a feature-based machine learning pipeline instead of relying on deep learning, making it computationally efficient and interpretable. Key steps include:

Frame Sampling: Extracts key frames at regular intervals to represent the entire video while minimizing processing load.

Visual Feature Extraction:

Motion Patterns (Optical Flow)

Color Statistics (mean, std, skewness, entropy)

Edge Detection (Canny, Sobel orientation)

Object & Texture Descriptors (HOG, ORB)

Metadata (frame rate, resolution, duration)

Model Training: A RandomForestClassifier is trained on these features with hyperparameter tuning to maximize accuracy while minimizing false positives.

Inference: Videos are scored for safety, assigned a risk level, and flagged accordingly.

CSV-based Results: Classifications and confidence scores are output in structured CSV files for easy integration and review.

⚙️ Technical Challenges & Solutions
🔄 Computational Efficiency
Processing every frame is resource-intensive. We optimized this by:

Using adaptive frame sampling

Parallelizing feature extraction using ProcessPoolExecutor

Caching features to avoid redundant computations

🎯 Reducing False Positives
False alarms can erode trust. To address this:

We added confidence thresholds

Implemented multi-feature voting across frames

Used class balancing during model training

🤔 Handling Ambiguity
Some frames are visually ambiguous. We:

Trained the model on diverse real-world samples

Emphasized motion and temporal features to capture context

📏 Scalability
Videos come in all sizes and formats. Our pipeline supports:

Robust handling of missing/corrupted frames

Scalable architecture that processes videos in parallel

🔐 Privacy & Security
To reduce human exposure to harmful visuals:

Only key information (e.g., classification and confidence) is shared

Visual previews are optional and protected

🌟 Why This System Stands Out
✅ Protects Moderators
Reduces psychological exposure to harmful content by automating the initial screening.

🚀 Scalable & Efficient
Handles thousands of videos without GPU dependency. New hardware scales performance linearly.

🎯 Nuanced Decisions
Provides risk scores and harm type labels, not just Safe/Unsafe flags—enabling better moderation decisions (e.g., restrict vs. remove).

📈 Transparent & Trustworthy
Every classification includes a confidence score, helping teams prioritize human review efficiently.

🔧 Easy to Integrate
Works out-of-the-box with content management systems via CSV mapping—ideal for platforms of any size.

🧩 Final Words
This project demonstrates how classical machine learning, when engineered thoughtfully, can offer powerful, transparent, and resource-efficient solutions to some of the toughest challenges in modern content moderation. It’s modular, privacy-conscious, and designed to scale—empowering digital platforms to create safer environments without sacrificing speed or accuracy.


  🚀 Performance Highlights
Our model demonstrates exceptional performance in video content moderation:

96% Overall Accuracy on the test dataset
100% Precision for unsafe content identification (zero false positives)
Near-Perfect AUC-ROC Score of 0.9996 showing excellent discrimination capability
Efficient Processing at ~1.02 videos per second on standard hardware
Highly Scalable with 1,753 videos successfully processed in a single batch

📊 Results Summary

![WhatsApp Image 2025-04-21 at 00 12 25_1931fcfd](https://github.com/user-attachments/assets/741a5d68-c6e9-4566-81c0-9e30853e9f5d)

The system achieves perfect precision for unsafe content identification, meaning when our model flags content as unsafe, it is consistently correct. This significantly reduces the risk of incorrectly restricting appropriate content.
💡 Technical Approach

Comprehensive Feature Extraction capturing nuanced patterns across video frames
Advanced Random Forest Classification with optimized hyperparameters
Class Imbalance Handling through balanced subsample weighting
Extensive GridSearch Cross-Validation for parameter tuning

The model identifies the most predictive features automatically, with the top features demonstrating significant discriminative power for content classification.
🛠️ Implementation Details

Data Processing: Robust pipeline handling 1,756 videos with minimal failures
Model Optimization: 16 candidate configurations evaluated across 5-fold cross-validation
Feature Importance Analysis: Clear identification of the most significant predictors
Fast Execution: Complete pipeline execution in under 30 minutes for large dataset
Ready-to-Use: Includes example inference code for seamless integration

🔧 Usage Example


![WhatsApp Image 2025-04-21 at 00 13 07_dddabf42](https://github.com/user-attachments/assets/29ddf25d-e34c-439f-9123-b320193f5600)

🔍 Future Improvements

Fine-tuning recall for unsafe content detection
Expanding feature extraction for better category classification
Implementing real-time processing capabilities
Adding visualization tools for detected content patterns


This project demonstrates the power of machine learning for content moderation, enabling platforms to create safer digital environments while reducing the burden on human moderators.
