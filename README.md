# Video Content Moderation

An automated system for classifying videos as **Safe** or **Unsafe** using classical machine learning. The system supports single-video inference, batch processing, and content-based trimming based on moderation results.

## Overview

Moderating video content at scale is one of the more difficult challenges in digital media, given the volume, format diversity, and sensitivity of the material involved. This project addresses that challenge with a fully automated, frame-level moderation pipeline that flags harmful content while minimizing manual review.

Rather than treating each video as a single unit, the system analyzes video frame-by-frame — typically sampling at one frame per second — so that inappropriate or harmful segments can be detected even when they occur briefly within a longer video. Each sampled frame is scored using handcrafted visual features and a trained classifier, producing a confidence-weighted **Safe / Unsafe** verdict for the video.

The pipeline is feature-based rather than deep-learning-based, which keeps it computationally efficient, interpretable, and deployable without GPU infrastructure.

## Performance

| Metric | Result |
|---|---|
| Overall accuracy | 96% |
| Precision (unsafe content) | 100% (zero false positives) |
| AUC-ROC | 0.9996 |
| Throughput | ~1.02 videos/second on standard hardware |
| Batch scale tested | 1,753 videos processed in a single run |

When the model flags content as unsafe, it is consistently correct, which minimizes the risk of incorrectly restricting appropriate content.

## How It Works

### 1. Train the Model
```bash
python train_model.py
```
- Reads video metadata and labels from `val.csv`
- Extracts visual features from each video
- Trains a `RandomForestClassifier` with hyperparameter tuning via grid search and 5-fold cross-validation
- Saves the trained model as `video_classifier_v2.pkl`

### 2. Classify a Single Video
```bash
python test_video_classifier.py --video path/to/video.mp4
```

### 3. Batch Process a Folder
```bash
python test_video_classifier.py --folder path/to/videos/
```
Results are written to a CSV file for downstream review or integration.

## Feature Extraction

The classifier is trained on the following visual and metadata features:

| Category | Features |
|---|---|
| Motion | Optical flow analysis |
| Color | Mean, standard deviation, skewness, entropy |
| Edges | Canny edge detection, Sobel orientation histogram |
| Texture / Objects | HOG (Histogram of Oriented Gradients), ORB keypoints and descriptors |
| Structure | Scene change rate |
| Metadata | Frame rate, resolution, duration |

## Dataset Format

Training data is expected in `val.csv` with the following columns:

| Column | Description |
|---|---|
| `Video_id` | Filename without extension |
| `Label` | `Safe` or `Unsafe` |
| `Category` | Category label used by utility scripts |

## Utility Scripts

| Script | Purpose |
|---|---|
| `categorynumcalc.py` | Counts the number of unique categories in the dataset |
| `get_duration.py` | Computes video durations for a given folder |
| `videoselectioncode.py` | Selects a fixed sample (5 videos) per category for analysis |

## Inference Example

```python
import joblib
from feature_extraction import extract_video_features

model = joblib.load("video_classifier_v2.pkl")
features = extract_video_features("path/to/video.mp4")
prediction = model.predict([features])
confidence = model.predict_proba([features])

print(f"Label: {prediction[0]}, Confidence: {confidence.max():.2f}")
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Tqdm

Install with:
```bash
pip install -r requirements.txt
```

## Technical Approach

**Frame sampling.** Key frames are extracted at regular intervals to represent the full video while keeping processing load manageable.

**Feature extraction.** Motion, color, edge, texture, and metadata features are computed per frame (see [Feature Extraction](#feature-extraction)).

**Model training.** A `RandomForestClassifier` is trained on the extracted features with hyperparameter tuning to maximize accuracy while minimizing false positives. Class imbalance is addressed through balanced subsample weighting, and 16 candidate configurations were evaluated across 5-fold cross-validation.

**Inference.** Each video receives a safety verdict, a risk level, and a confidence score, output in structured CSV form for easy integration into existing review workflows.

## Design Considerations

**Computational efficiency.** Processing every frame is resource-intensive, so the pipeline uses adaptive frame sampling, parallelizes feature extraction with `ProcessPoolExecutor`, and caches features to avoid redundant computation.

**False positive control.** Confidence thresholds, multi-frame voting, and class balancing during training work together to reduce false alarms, which is reflected in the model's 100% precision on unsafe content.

**Ambiguity handling.** The model is trained on diverse real-world samples and weights motion and temporal features to better capture context in visually ambiguous frames.

**Scalability.** The pipeline handles missing or corrupted frames gracefully and processes videos in parallel, allowing it to scale across large batches without GPU dependency.

**Privacy.** Only classification results and confidence scores are surfaced by default; visual previews are optional, reducing moderator exposure to harmful content.

## Why This Approach

- **Protects moderators** by automating the initial screening pass, reducing psychological exposure to harmful content.
- **Scales without GPUs** — throughput scales roughly linearly with additional hardware.
- **Delivers nuanced decisions** — risk scores and harm-type labels support graduated responses (e.g., restrict vs. remove) rather than a binary flag alone.
- **Is transparent** — every classification includes a confidence score, helping teams prioritize human review.
- **Integrates easily** — CSV-based input/output works with most existing content management workflows.

## Future Improvements

- Improve recall for unsafe content detection
- Expand feature extraction to support finer-grained category classification
- Add real-time processing support
- Build visualization tooling for detected content patterns

## License

This project is licensed under the MIT License. See `LICENSE` for details.
