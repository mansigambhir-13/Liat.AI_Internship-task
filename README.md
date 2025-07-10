# \# Player Re-identification in Sports Footage

# 

# \*\*Assignment\*\*: Liat.ai AI Intern - Player Re-Identification in Sports Footage  

# \*\*Option Implemented\*\*: Option 2 - Re-Identification in a Single Feed  

# \*\*Submission Date\*\*: July 2025

# 

# \## Overview

# 

# This project implements a computer vision solution for player re-identification in sports footage, specifically addressing \*\*Option 2: Re-Identification in a Single Feed\*\* from the Liat.ai assignment. The system tracks players throughout a video sequence and maintains consistent player IDs even when players temporarily leave the frame and reappear later.

# 

# \## Key Features

# 

# \- \*\*YOLOv11-based Player Detection\*\*: Uses the provided fine-tuned YOLOv11 model for accurate player detection

# \- \*\*Multi-feature Re-identification\*\*: Combines color, texture, shape, and spatial features for robust player identification

# \- \*\*Hungarian Algorithm Assignment\*\*: Optimal detection-to-track association using the Hungarian algorithm

# \- \*\*Temporal Tracking\*\*: Maintains player tracks across frames with motion prediction

# \- \*\*Re-identification System\*\*: Automatically re-identifies players when they return to frame

# \- \*\*Real-time Simulation\*\*: Processes video frame-by-frame to simulate real-time scenarios

# 

# \## Project Structure

# 

# ```

# player\_reidentification/

# ├── src/

# │   ├── detector.py              # YOLOv11 player detection module

# │   ├── feature\_extractor.py     # Multi-feature extraction for re-identification

# │   ├── reid\_system.py          # Main tracking and re-identification system

# │   └── main.py                 # Main execution script

# ├── data/

# │   ├── videos/

# │   │   └── 15sec\_input\_720p.mp4  # Input video file

# │   └── models/

# │       └── yolov11\_player\_detection.pt  # Pre-trained model

# ├── outputs/

# │   ├── final\_results.json      # Frame-by-frame tracking results

# │   └── tracked\_video.mp4       # Output video with tracking visualization (optional)

# ├── reid\_env/                   # Virtual environment

# ├── README.md                   # This file

# ├── report.md                   # Technical report

# ├── requirements\_final.txt      # Python dependencies

# └── SUBMISSION\_SUMMARY.md       # Submission overview

# ```

# 

# \## Dependencies and Environment Requirements

# 

# \### System Requirements

# \- \*\*Python\*\*: 3.8 or higher

# \- \*\*Operating System\*\*: Windows, macOS, or Linux

# \- \*\*Memory\*\*: 4GB+ RAM recommended

# \- \*\*Storage\*\*: 2GB+ free space

# 

# \### Python Dependencies

# ```

# torch>=1.13.0

# torchvision>=0.14.0

# ultralytics>=8.0.0

# opencv-python>=4.7.0

# pillow>=9.0.0

# scikit-image>=0.19.0

# numpy>=1.21.0

# scipy>=1.8.0

# scikit-learn>=1.1.0

# matplotlib>=3.5.0

# ```

# 

# \## Installation and Setup

# 

# \### Step 1: Clone/Download the Repository

# ```bash

# git clone <your-repo-url>

# cd player\_reidentification

# ```

# 

# \### Step 2: Create Virtual Environment

# ```bash

# \# Create virtual environment

# python -m venv reid\_env

# 

# \# Activate environment

# \# On Windows:

# reid\_env\\Scripts\\activate

# \# On Linux/Mac:

# source reid\_env/bin/activate

# ```

# 

# \### Step 3: Install Dependencies

# ```bash

# \# Install all required packages

# pip install -r requirements\_final.txt

# 

# \# Verify installations

# python -c "import torch; print('PyTorch:', torch.\_\_version\_\_)"

# python -c "import cv2; print('OpenCV:', cv2.\_\_version\_\_)"

# python -c "import ultralytics; print('Ultralytics installed')"

# ```

# 

# \### Step 4: Download Required Files

# 1\. \*\*YOLOv11 Model\*\*: Download from the provided Google Drive link and place in `data/models/yolov11\_player\_detection.pt`

# 2\. \*\*Video Data\*\*: Download `15sec\_input\_720p.mp4` and place in `data/videos/`

# 

# \## How to Set Up and Run the Code

# 

# \### Basic Usage

# ```bash

# \# Ensure virtual environment is activated

# reid\_env\\Scripts\\activate  # Windows

# \# source reid\_env/bin/activate  # Linux/Mac

# 

# \# Run basic tracking (JSON output only)

# python src\\main.py --video data\\videos\\15sec\_input\_720p.mp4 --model data\\models\\yolov11\_player\_detection.pt --output\_json outputs\\results.json

# ```

# 

# \### Advanced Usage with All Options

# ```bash

# python src\\main.py \\

# &nbsp;   --video data\\videos\\15sec\_input\_720p.mp4 \\

# &nbsp;   --model data\\models\\yolov11\_player\_detection.pt \\

# &nbsp;   --output\_video outputs\\tracked\_video.mp4 \\

# &nbsp;   --output\_json outputs\\tracking\_results.json \\

# &nbsp;   --similarity\_threshold 0.3 \\

# &nbsp;   --reid\_threshold 0.4 \\

# &nbsp;   --max\_inactive\_frames 30

# ```

# 

# \### Command Line Parameters

# \- `--video`: Path to input video file (required)

# \- `--model`: Path to YOLOv11 model file (required)

# \- `--output\_video`: Path to save annotated output video (optional)

# \- `--output\_json`: Path to save tracking results JSON (default: tracking\_results.json)

# \- `--similarity\_threshold`: Threshold for track association (default: 0.3)

# \- `--reid\_threshold`: Threshold for re-identification (default: 0.4)

# \- `--max\_inactive\_frames`: Max frames before removing inactive tracks (default: 30)

# 

# \## Output Description

# 

# \### 1. Tracking Results JSON

# The system generates a JSON file containing frame-by-frame tracking data:

# ```json

# {

# &nbsp; "0": \[

# &nbsp;   {

# &nbsp;     "track\_id": 1,

# &nbsp;     "bbox": \[100, 150, 180, 300],

# &nbsp;     "confidence": 0.87,

# &nbsp;     "frames\_tracked": 45,

# &nbsp;     "active": true

# &nbsp;   }

# &nbsp; ],

# &nbsp; "1": \[ ... ],

# &nbsp; ...

# }

# ```

# 

# \### 2. Console Output

# The system provides real-time progress and final statistics:

# ```

# Processing video: data/videos/15sec\_input\_720p.mp4

# Processed frame 0

# Processed frame 30

# ...

# Processing complete. Total frames: 375

# 

# ==================================================

# TRACKING STATISTICS

# ==================================================

# Total tracks created: 21

# Currently active tracks: 17

# Inactive tracks: 4

# Frames processed: 375

# ```

# 

# \### 3. Optional Annotated Video

# When `--output\_video` is specified, creates an MP4 file with:

# \- Bounding boxes around detected players

# \- Track IDs displayed above each player

# \- Color-coded tracks for easy visualization

# 

# \## Expected Performance

# 

# Based on the provided test video:

# \- \*\*Detection Rate\*\*: ~16-17 players per frame

# \- \*\*Tracking Consistency\*\*: 21 unique tracks maintained

# \- \*\*Processing Speed\*\*: 2-5 minutes for 15-second video

# \- \*\*Re-identification Success\*\*: High accuracy when players return to frame

# 

# \## Troubleshooting

# 

# \### Common Issues and Solutions

# 

# 1\. \*\*Model Loading Error\*\*

# &nbsp;  ```

# &nbsp;  Error: Model file not found

# &nbsp;  ```

# &nbsp;  - Solution: Ensure YOLOv11 model is downloaded and placed in correct directory

# 

# 2\. \*\*Video Loading Error\*\*

# &nbsp;  ```

# &nbsp;  Error: Video file not found

# &nbsp;  ```

# &nbsp;  - Solution: Verify video file path and format compatibility

# 

# 3\. \*\*Import Errors\*\*

# &nbsp;  ```

# &nbsp;  ModuleNotFoundError: No module named 'cv2'

# &nbsp;  ```

# &nbsp;  - Solution: Ensure virtual environment is activated and dependencies installed

# 

# 4\. \*\*Memory Issues\*\*

# &nbsp;  ```

# &nbsp;  Out of memory error

# &nbsp;  ```

# &nbsp;  - Solution: Close other applications or reduce video resolution

# 

# 5\. \*\*Low Performance\*\*

# &nbsp;  - Solution: Adjust similarity thresholds or reduce max\_inactive\_frames

# 

# \### Debug Mode

# For troubleshooting, you can test individual components:

# ```bash

# \# Test detector only

# python -c "

# import sys; sys.path.append('src')

# from detector import PlayerDetector

# detector = PlayerDetector('data/models/yolov11\_player\_detection.pt')

# print('Detector loaded successfully')

# "

# 

# \# Test video loading

# python -c "

# import cv2

# cap = cv2.VideoCapture('data/videos/15sec\_input\_720p.mp4')

# print('Video opened:', cap.isOpened())

# cap.release()

# "

# ```

# 

# \## Algorithm Overview

# 

# \### Detection Pipeline

# 1\. \*\*YOLOv11 Detection\*\*: Identifies players in each frame (class\_id=2)

# 2\. \*\*Feature Extraction\*\*: Extracts multi-modal features for each detection

# 3\. \*\*Track Association\*\*: Uses Hungarian algorithm for optimal assignment

# 4\. \*\*Re-identification\*\*: Matches returning players with inactive tracks

# 

# \### Key Technical Components

# \- \*\*Multi-feature Similarity\*\*: Combines color histograms, HOG, LBP, and spatial features

# \- \*\*Motion Prediction\*\*: Linear velocity model for position prediction

# \- \*\*Track Management\*\*: Smart activation/deactivation of player tracks

# \- \*\*Robust Matching\*\*: Weighted similarity scoring with reliability boosting

# 

# \## Citation and References

# 

# This implementation uses:

# \- \*\*YOLOv11\*\*: Ultralytics implementation for object detection

# \- \*\*Hungarian Algorithm\*\*: SciPy's linear\_sum\_assignment for optimal matching

# \- \*\*HOG Features\*\*: Scikit-image's histogram of oriented gradients

# \- \*\*OpenCV\*\*: Computer vision operations and video processing

# 

# \## License

# 

# This project is developed for the Liat.ai assignment and is intended for educational and evaluation purposes.

# 

# \## Contact

# 

# For questions or issues regarding this implementation, please contact the assignment evaluators at Liat.ai.

