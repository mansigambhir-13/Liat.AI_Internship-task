# detector.py - Fixed version
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class PlayerDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect_players(self, frame):
        results = self.model(frame, device=self.device, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    # Look for class_id == 2 (players), not 0!
                    if confidence >= self.confidence_threshold and class_id == 2:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        player_crop = frame[y1:y2, x1:x2]
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'player_crop': player_crop,
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                        detections.append(detection)
        
        return detections
    
    def detect_video(self, video_path, output_dir=None):
        cap = cv2.VideoCapture(video_path)
        frame_detections = {}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.detect_players(frame)
            frame_detections[frame_idx] = detections
            
            if output_dir:
                annotated_frame = self.draw_detections(frame, detections)
                cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", annotated_frame)
            
            frame_idx += 1
            
        cap.release()
        return frame_detections
    
    def draw_detections(self, frame, detections):
        annotated_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Player {i}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
