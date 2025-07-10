# reid_system.py
import numpy as np
import cv2
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from detector import PlayerDetector
from feature_extractor import FeatureExtractor

class PlayerTrack:
    """Represents a single player track"""
    def __init__(self, track_id, initial_detection, initial_features, frame_idx):
        self.track_id = track_id
        self.bbox_history = deque(maxlen=30)  # Keep last 30 bboxes
        self.feature_history = deque(maxlen=10)  # Keep last 10 feature sets
        self.confidence_history = deque(maxlen=10)
        
        # State variables
        self.active = True
        self.frames_since_update = 0
        self.total_frames_tracked = 1
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        
        # Motion prediction (simple linear model)
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.position = np.array([
            (initial_detection['bbox'][0] + initial_detection['bbox'][2]) / 2,
            (initial_detection['bbox'][1] + initial_detection['bbox'][3]) / 2
        ])
        
        # Add initial data
        self.update(initial_detection, initial_features, frame_idx)
    
    def update(self, detection, features, frame_idx):
        """Update track with new detection"""
        self.bbox_history.append(detection['bbox'])
        self.feature_history.append(features)
        self.confidence_history.append(detection['confidence'])
        
        self.frames_since_update = 0
        self.last_seen_frame = frame_idx
        self.total_frames_tracked += 1
        
        # Update position and velocity
        new_center = np.array([detection['center'][0], detection['center'][1]])
        if len(self.bbox_history) > 1:
            self.velocity = new_center - self.position
        self.position = new_center
    
    def predict_next_position(self):
        """Predict next position based on current velocity"""
        return self.position + self.velocity
    
    def get_current_features(self):
        """Get the most representative features"""
        if not self.feature_history:
            return None
        
        # For now, return the most recent features
        # Could be improved by averaging recent features
        return self.feature_history[-1]
    
    def get_average_features(self, n_recent=3):
        """Get averaged features from recent detections"""
        if not self.feature_history:
            return None
        
        recent_features = list(self.feature_history)[-n_recent:]
        if not recent_features:
            return None
        
        # Average numerical features
        avg_features = {}
        
        # For histogram features, average them
        histogram_keys = ['color_hist_h', 'color_hist_s', 'color_hist_v', 'lbp_hist']
        for key in histogram_keys:
            if key in recent_features[0]:
                avg_features[key] = np.mean([f[key] for f in recent_features], axis=0)
        
        # For color features, average them
        color_keys = ['jersey_color', 'shorts_color']
        for key in color_keys:
            if key in recent_features[0]:
                avg_features[key] = np.mean([f[key] for f in recent_features], axis=0)
        
        # For HOG features, average them
        if 'hog_features' in recent_features[0]:
            avg_features['hog_features'] = np.mean([f['hog_features'] for f in recent_features], axis=0)
        
        # For shape features, use the most recent
        shape_keys = ['bbox_width', 'bbox_height', 'aspect_ratio', 'bbox_area']
        for key in shape_keys:
            if key in recent_features[-1]:
                avg_features[key] = recent_features[-1][key]
        
        return avg_features
    
    def mark_inactive(self, frame_idx):
        """Mark track as inactive"""
        self.active = False
        self.frames_since_update = frame_idx - self.last_seen_frame

class PlayerReIdentificationSystem:
    """Main re-identification system"""
    
    def __init__(self, model_path, similarity_threshold=0.3, max_frames_inactive=30):
        self.detector = PlayerDetector(model_path)
        self.feature_extractor = FeatureExtractor()
        
        # Tracking parameters
        self.similarity_threshold = similarity_threshold
        self.max_frames_inactive = max_frames_inactive
        self.position_weight = 0.3  # Weight for spatial proximity in assignment
        
        # Track management
        self.active_tracks = {}
        self.inactive_tracks = {}
        self.next_track_id = 1
        self.frame_idx = 0
        
        # For re-identification
        self.reid_similarity_threshold = 0.4
        
    def process_frame(self, frame):
        """Process a single frame and return tracking results"""
        # Detect players in current frame
        detections = self.detector.detect_players(frame)
        
        # Extract features for all detections
        detection_features = []
        for detection in detections:
            features = self.feature_extractor.extract_features(
                detection['player_crop'], 
                detection['bbox'],
                {'height': frame.shape[0], 'width': frame.shape[1]}
            )
            detection_features.append(features)
        
        # Associate detections with existing tracks
        assignments = self._associate_detections(detections, detection_features)
        
        # Update tracks and handle new/lost tracks
        self._update_tracks(detections, detection_features, assignments)
        
        # Increment frame counter
        self.frame_idx += 1
        
        # Return current tracking results
        return self._get_current_results()
    
    def _associate_detections(self, detections, detection_features):
        """Associate detections with existing tracks using Hungarian algorithm"""
        if not self.active_tracks or not detections:
            return {}
        
        track_ids = list(self.active_tracks.keys())
        n_tracks = len(track_ids)
        n_detections = len(detections)
        
        # Create cost matrix
        cost_matrix = np.full((n_tracks, n_detections), 1.0)  # High cost = low similarity
        
        for i, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            track_features = track.get_average_features()
            predicted_pos = track.predict_next_position()
            
            for j, (detection, det_features) in enumerate(zip(detections, detection_features)):
                # Calculate feature similarity
                feature_sim = 0.0
                if track_features:
                    feature_sim = self.feature_extractor.calculate_similarity(track_features, det_features)
                
                # Calculate spatial proximity
                det_center = np.array(detection['center'])
                spatial_dist = np.linalg.norm(predicted_pos - det_center)
                # Normalize spatial distance (assuming max reasonable distance is 200 pixels)
                spatial_sim = np.exp(-spatial_dist / 100.0)
                
                # Combined similarity
                combined_sim = (1 - self.position_weight) * feature_sim + self.position_weight * spatial_sim
                
                # Convert to cost (1 - similarity)
                cost_matrix[i, j] = 1.0 - combined_sim
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter assignments by similarity threshold
        assignments = {}
        for row, col in zip(row_indices, col_indices):
            similarity = 1.0 - cost_matrix[row, col]
            if similarity >= self.similarity_threshold:
                track_id = track_ids[row]
                assignments[track_id] = col
        
        return assignments
    
    def _update_tracks(self, detections, detection_features, assignments):
        """Update tracks based on assignments and handle new/lost tracks"""
        # Update assigned tracks
        assigned_detection_indices = set(assignments.values())
        
        for track_id, detection_idx in assignments.items():
            detection = detections[detection_idx]
            features = detection_features[detection_idx]
            self.active_tracks[track_id].update(detection, features, self.frame_idx)
        
        # Mark unassigned tracks as having missed detection
        for track_id in self.active_tracks:
            if track_id not in assignments:
                self.active_tracks[track_id].frames_since_update += 1
        
        # Move inactive tracks to inactive list
        tracks_to_deactivate = []
        for track_id, track in self.active_tracks.items():
            if track.frames_since_update > self.max_frames_inactive:
                tracks_to_deactivate.append(track_id)
        
        for track_id in tracks_to_deactivate:
            track = self.active_tracks.pop(track_id)
            track.mark_inactive(self.frame_idx)
            self.inactive_tracks[track_id] = track
        
        # Handle unassigned detections (potential new tracks or re-identifications)
        unassigned_detections = []
        for i, (detection, features) in enumerate(zip(detections, detection_features)):
            if i not in assigned_detection_indices:
                unassigned_detections.append((detection, features))
        
        # Try to re-identify with inactive tracks first
        for detection, features in unassigned_detections:
            reid_track_id = self._attempt_reidentification(detection, features)
            if reid_track_id:
                # Reactivate the track
                track = self.inactive_tracks.pop(reid_track_id)
                track.active = True
                track.update(detection, features, self.frame_idx)
                self.active_tracks[reid_track_id] = track
            else:
                # Create new track
                self._create_new_track(detection, features)
    
    def _attempt_reidentification(self, detection, features):
        """Attempt to re-identify detection with inactive tracks"""
        best_similarity = 0.0
        best_track_id = None
        
        for track_id, track in self.inactive_tracks.items():
            # Only consider tracks that were recently active
            if self.frame_idx - track.last_seen_frame > self.max_frames_inactive * 2:
                continue
            
            track_features = track.get_average_features()
            if track_features:
                similarity = self.feature_extractor.calculate_similarity(track_features, features)
                
                # Boost similarity for tracks that were more reliably tracked
                reliability_boost = min(track.total_frames_tracked / 30.0, 0.2)  # Max 0.2 boost
                adjusted_similarity = similarity + reliability_boost
                
                if adjusted_similarity > best_similarity and adjusted_similarity >= self.reid_similarity_threshold:
                    best_similarity = adjusted_similarity
                    best_track_id = track_id
        
        return best_track_id
    
    def _create_new_track(self, detection, features):
        """Create a new track for unassigned detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        new_track = PlayerTrack(track_id, detection, features, self.frame_idx)
        self.active_tracks[track_id] = new_track
        
        return track_id
    
    def _get_current_results(self):
        """Get current tracking results for visualization/output"""
        results = []
        
        for track_id, track in self.active_tracks.items():
            if track.bbox_history:
                bbox = track.bbox_history[-1]
                confidence = track.confidence_history[-1] if track.confidence_history else 0.0
                
                result = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'frames_tracked': track.total_frames_tracked,
                    'active': track.active
                }
                results.append(result)
        
        return results
    
    def process_video(self, video_path, output_path=None):
        """Process entire video and return all tracking results"""
        cap = cv2.VideoCapture(video_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for output
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = {}
        frame_idx = 0
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            all_results[frame_idx] = results
            
            # Draw results and save frame
            if output_path:
                annotated_frame = self._draw_tracking_results(frame, results)
                out.write(annotated_frame)
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}")
            
            frame_idx += 1
        
        cap.release()
        if out:
            out.release()
        
        print(f"Processing complete. Total frames: {frame_idx}")
        return all_results
    
    def _draw_tracking_results(self, frame, results):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        # Color palette for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0)
        ]
        
        for result in results:
            track_id = result['track_id']
            bbox = result['bbox']
            confidence = result['confidence']
            
            # Choose color based on track_id
            color = colors[track_id % len(colors)]
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            label = f"ID:{track_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_track_statistics(self):
        """Get statistics about tracking performance"""
        stats = {
            'total_tracks_created': self.next_track_id - 1,
            'currently_active_tracks': len(self.active_tracks),
            'inactive_tracks': len(self.inactive_tracks),
            'frames_processed': self.frame_idx
        }
        
        # Per-track statistics
        track_stats = []
        for track_id, track in {**self.active_tracks, **self.inactive_tracks}.items():
            track_stat = {
                'track_id': track_id,
                'total_frames_tracked': track.total_frames_tracked,
                'first_seen_frame': track.first_seen_frame,
                'last_seen_frame': track.last_seen_frame,
                'active': track.active
            }
            track_stats.append(track_stat)
        
        stats['track_details'] = track_stats
        return stats
