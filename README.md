# Player Re-identification in Sports Footage

This project implements a computer vision solution for player re-identification in sports footage, specifically addressing **Option 2: Re-Identification in a Single Feed** from the Liat.ai assignment.

## Overview

The system tracks players throughout a video sequence and maintains consistent player IDs even when players temporarily leave the frame and reappear later. This simulates real-time player tracking and re-identification scenarios common in sports analytics.

## Key Features

- **YOLOv11-based Player Detection**: Uses the provided fine-tuned YOLOv11 model for accurate player detection
- **Multi-feature Re-identification**: Combines color, texture, shape, and spatial features for robust player identification
- **Hungarian Algorithm Assignment**: Optimal detection-to-track association using the Hungarian algorithm
- **Temporal Tracking**: Maintains player tracks across frames with motion prediction
- **Re-identification System**: Automatically re-identifies players when they return to frame
- **Real-time Simulation**: Processes video frame-by-frame to simulate real-time scenarios

## Project Structure

```
player_reidentification/
├── src/
│   ├── detector.py              # YOLOv11 player detection module
│   ├── feature_extractor.py     # Multi-feature extraction for re-identification
│   ├── reid_system.py          # Main tracking and re-identification system
│   └── main.py                 # Main execution script
├── data/
│   ├── videos/
│   │   └── 15sec_input_720p.mp4  # Input video file
│   └── models/
│       └── yolov11_player_detection.pt  # Pre-trained model
├── outputs/
│   ├── tracking_results.json    # Frame-by-frame tracking results
│   └── annotated_video.mp4     # Output video with tracking visualization
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── report.md                  # Technical report
```

## Installation and Setup

### 1. Clone/Download the Repository

```bash
git clone <your-repo-url>
cd player_reidentification
```

### 2. Create Virtual Environment

```bash
python -m venv reid_env
source reid_env/bin/activate  # On Windows: reid_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Files

1. **Download the YOLOv11 model** from the provided Google Drive link:
   - Place it in `data/models/yolov11_player_detection.pt`

2. **Download the test video** from the assignment materials:
   - Place it in `data/videos/15sec_input_720p.mp4`

## Usage

### Basic Usage

```bash
python src/main.py --video data/videos/15sec_input_720p.mp4 --model data/models/yolov11_player_detection.pt
```

### Advanced Usage with Custom Parameters

```bash
python src/main.py \
    --video data/videos/15sec_input_720p.mp4 \
    --model data/models/yolov11_player_detection.pt \
    --output_video outputs/tracked_video.mp4 \
    --output_json outputs/tracking_results.json \
    --similarity_threshold 0.3 \
    --reid_threshold 0.4 \
    --max_inactive_frames 30
```

### Parameters

- `--video`: Path to input video file (required)
- `--model`: Path to YOLOv11 model file (required)
- `--output_video`: Path to save annotated output video (optional)
- `--output_json`: Path to save tracking results JSON (default: tracking_results.json)
- `--similarity_threshold`: Threshold for track association (default: 0.3)
- `--reid_threshold`: Threshold for re-identification (default: 0.4)
- `--max_inactive_frames`: Max frames before removing inactive tracks (default: 30)

## Output

The system generates two main outputs:

### 1. Tracking Results JSON
Contains frame-by-frame tracking data:
```json
{
  "0": [
    {
      "track_id": 1,
      "bbox": [100, 150, 180, 300],
      "confidence": 0.87,
      "frames_tracked": 45,
      "active": true
    }
  ]
}
```

### 2. Annotated Video (Optional)
Video file with bounding boxes and track IDs overlaid on each frame.

## Technical Approach

### 1. Player Detection
- Uses fine-tuned YOLOv11 model for robust player detection
- Filters detections by confidence threshold
- Extracts player bounding boxes and cropped regions

### 2. Feature Extraction
Combines multiple feature types for robust re-identification:

**Color Features:**
- HSV color histograms (more robust to lighting changes)
- Dominant jersey and shorts colors
- Color moments (mean, std, skewness)

**Texture Features:**
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)

**Shape Features:**
- Bounding box dimensions and aspect ratio
- Relative size compared to frame

**Spatial Features:**
- Normalized position in frame
- Motion vectors and velocity

### 3. Tracking and Association
- **Hungarian Algorithm**: Optimal assignment of detections to tracks
- **Motion Prediction**: Linear motion model for predicting next positions
- **Multi-criteria Similarity**: Combines feature similarity and spatial proximity

### 4. Re-identification
- Maintains feature profiles for inactive tracks
- Compares new detections with inactive track features
- Reactivates best matching tracks above threshold

### 5. Track Management
- Creates new tracks for unassigned detections
- Deactivates tracks after extended periods without detection
- Maintains track history for improved re-identification

## Dependencies

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLOv11 implementation
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **SciPy**: Optimization algorithms (Hungarian method)
- **scikit-learn**: Machine learning utilities
- **scikit-image**: Image processing (HOG features)

## Performance Considerations

### Accuracy Optimizations
- Multiple feature types reduce false associations
- Temporal consistency through motion prediction
- Re-identification prevents ID switching for returning players

### Efficiency Optimizations
- Feature vector caching for active tracks
- Limited history storage (configurable)
- Early termination for low-confidence detections

## Evaluation

The system is evaluated on:
- **Accuracy**: Consistency of player IDs throughout video
- **Re-identification Success**: Correctly identifying returning players
- **False Positive Rate**: Incorrectly creating new tracks for existing players
- **Tracking Stability**: Maintaining tracks through occlusions and motion

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure YOLOv11 model is downloaded and placed correctly
2. **Video codec issues**: Install additional codecs if video won't load
3. **Memory issues**: Reduce history sizes in track configuration
4. **Poor tracking**: Adjust similarity thresholds based on video characteristics

### Debug Mode

For single-frame testing:
```python
# Uncomment in main.py
test_single_frame()
```

## Future Improvements

- **Deep Learning Features**: Integration of CNN-based re-identification features
- **Multi-camera Support**: Extension to cross-camera player mapping
- **Real-time Processing**: Optimization for live video streams
- **Team Classification**: Automatic team assignment based on jersey colors
- **Pose Features**: Integration of player pose/skeleton features

## License

This project is developed for the Liat.ai assignment and is intended for educational and evaluation purposes.
