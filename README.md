# video-content-moderation😃
🎥 AI-Powered Video Content Moderation & Trimming
An automated system for classifying videos as Safe or Unsafe using machine learning, with support for batch analysis and trimming based on content moderation results.

🚀 How to Use
1. Train the Model
Run train_model.py.

It reads video metadata and labels from val.csv
Extracts various video features
Trains a RandomForestClassifier with hyperparameter tuning
Saves the model as video_classifier_v2.pkl

2. Analyze a Single Video
Use test_video_classifier.py with the --video argument to classify a single video using the saved model.

3. Batch Analyze a Folder
Use test_video_classifier.py with the --folder argument to process all videos in a folder and save the results in a CSV file.

🧠 Feature Extraction
The system extracts these features from videos:
HOG (Histogram of Oriented Gradients)
ORB (Keypoints and Descriptors)
Motion analysis using Optical Flow
Color statistics: mean, standard deviation, skewness, entropy
Edge features: Canny edge and Sobel histogram
Scene change rate

Metadata: fps, resolution, duration, etc.

📊 Dataset Format
Data should be stored in val.csv with the following columns:

Video_id: filename without extension

Label: Safe or Unsafe

Category: category label used in utility scripts

📌 Utilities
categorynumcalc.py: Counts the number of unique categories
get_duration.py: Computes video durations in a folder
videoselectioncode.py: Selects 5 videos per category for analysis

✅ Inference Example
To make predictions:
Load the model with joblib
Extract features from a video using extract_video_features
Use the model to predict the label and confidence
Output: "Safe" or "Unsafe" along with a confidence score

📦 Requirements
Python 3.8 or higher
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
