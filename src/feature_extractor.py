# feature_extractor.py - Fixed version
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with default parameters"""
        self.color_bins = 32  # Number of bins for color histograms
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
    def extract_features(self, player_crop, bbox=None, frame_info=None):
        """
        Extract comprehensive features from player crop
        """
        if player_crop.size == 0:
            return self._get_empty_features()
        
        features = {}
        
        # 1. Color features
        features.update(self._extract_color_features(player_crop))
        
        # 2. Texture features
        features.update(self._extract_texture_features(player_crop))
        
        # 3. Shape features FIRST
        features.update(self._extract_shape_features(player_crop, bbox))
        
        # 4. Spatial features AFTER shape features
        if bbox and frame_info:
            features.update(self._extract_spatial_features(bbox, frame_info, features))
        
        return features
    
    def _extract_color_features(self, player_crop):
        """Extract color-based features"""
        features = {}
        
        # Resize crop for consistent processing
        crop_resized = cv2.resize(player_crop, (64, 128))
        
        # HSV color histogram
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        
        # Full image histogram
        hist_h = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
        
        features['color_hist_h'] = hist_h
        features['color_hist_s'] = hist_s
        features['color_hist_v'] = hist_v
        
        # Dominant colors
        h, w = crop_resized.shape[:2]
        upper_region = hsv[:h//2, :]  # Upper body (jersey)
        lower_region = hsv[h//2:, :]  # Lower body (shorts/legs)
        
        features['jersey_color'] = self._get_dominant_color(upper_region)
        features['shorts_color'] = self._get_dominant_color(lower_region)
        
        # Color moments
        features.update(self._calculate_color_moments(hsv))
        
        return features
    
    def _extract_texture_features(self, player_crop):
        """Extract texture-based features using HOG"""
        features = {}
        
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (64, 128))
        
        # Extract HOG features
        try:
            hog_features = hog(gray_resized, 
                             orientations=self.hog_orientations,
                             pixels_per_cell=self.hog_pixels_per_cell,
                             cells_per_block=self.hog_cells_per_block,
                             block_norm='L2-Hys')
            features['hog_features'] = hog_features
        except:
            features['hog_features'] = np.zeros(1764)  # Default HOG size
        
        # Local Binary Pattern (LBP) for texture
        features['lbp_hist'] = self._calculate_lbp(gray_resized)
        
        return features
    
    def _extract_shape_features(self, player_crop, bbox):
        """Extract shape and geometric features"""
        features = {}
        
        if bbox:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            features['bbox_width'] = width
            features['bbox_height'] = height
            features['aspect_ratio'] = width / (height + 1e-7)
            features['bbox_area'] = width * height
        else:
            h, w = player_crop.shape[:2]
            features['bbox_width'] = w
            features['bbox_height'] = h
            features['aspect_ratio'] = w / (h + 1e-7)
            features['bbox_area'] = w * h
        
        return features
    
    def _extract_spatial_features(self, bbox, frame_info, existing_features):
        """Extract spatial position features"""
        features = {}
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize by frame dimensions
        frame_height = frame_info.get('height', 720)
        frame_width = frame_info.get('width', 1280)
        
        features['normalized_center_x'] = center_x / frame_width
        features['normalized_center_y'] = center_y / frame_height
        
        # Use bbox_area from existing_features
        bbox_area = existing_features.get('bbox_area', (x2-x1)*(y2-y1))
        features['relative_size'] = bbox_area / (frame_width * frame_height)
        
        return features
    
    def _get_dominant_color(self, region):
        """Get dominant color from a region"""
        pixels = region.reshape(-1, 3)
        
        # Remove very dark pixels (likely shadows)
        bright_pixels = pixels[pixels[:, 2] > 30]  # V channel > 30
        
        if len(bright_pixels) == 0:
            return np.array([0, 0, 0])
        
        dominant_color = np.mean(bright_pixels, axis=0)
        return dominant_color
    
    def _calculate_color_moments(self, hsv_image):
        """Calculate color moments (mean, std, skewness)"""
        features = {}
        
        for i, channel in enumerate(['h', 's', 'v']):
            channel_data = hsv_image[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_skew'] = self._skewness(channel_data)
        
        return features
    
    def _calculate_lbp(self, gray_image):
        """Calculate Local Binary Pattern histogram"""
        h, w = gray_image.shape
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                code = 0
                code |= (gray_image[i-1, j-1] >= center) << 7
                code |= (gray_image[i-1, j] >= center) << 6
                code |= (gray_image[i-1, j+1] >= center) << 5
                code |= (gray_image[i, j+1] >= center) << 4
                code |= (gray_image[i+1, j+1] >= center) << 3
                code |= (gray_image[i+1, j] >= center) << 2
                code |= (gray_image[i+1, j-1] >= center) << 1
                code |= (gray_image[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def _skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _get_empty_features(self):
        """Return empty feature set when player crop is invalid"""
        return {
            'color_hist_h': np.zeros(self.color_bins),
            'color_hist_s': np.zeros(self.color_bins),
            'color_hist_v': np.zeros(self.color_bins),
            'jersey_color': np.array([0, 0, 0]),
            'shorts_color': np.array([0, 0, 0]),
            'hog_features': np.zeros(1764),
            'lbp_hist': np.zeros(256),
            'bbox_width': 0,
            'bbox_height': 0,
            'aspect_ratio': 1.0,
            'bbox_area': 0
        }
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        similarities = {}
        
        # Color histogram similarities
        for hist_type in ['color_hist_h', 'color_hist_s', 'color_hist_v']:
            if hist_type in features1 and hist_type in features2:
                hist_sim = np.sum(np.minimum(features1[hist_type], features2[hist_type]))
                similarities[hist_type] = hist_sim
        
        # Dominant color similarities
        for color_type in ['jersey_color', 'shorts_color']:
            if color_type in features1 and color_type in features2:
                dist = np.linalg.norm(features1[color_type] - features2[color_type])
                similarities[color_type] = np.exp(-dist / 50.0)
        
        # HOG feature similarity
        if 'hog_features' in features1 and 'hog_features' in features2:
            try:
                hog_sim = 1 - cosine(features1['hog_features'], features2['hog_features'])
                similarities['hog'] = max(0, hog_sim)
            except:
                similarities['hog'] = 0.0
        
        # Shape similarity
        if all(k in features1 and k in features2 for k in ['aspect_ratio', 'bbox_area']):
            ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
            ar_sim = np.exp(-ar_diff)
            
            area_ratio = min(features1['bbox_area'], features2['bbox_area']) / (max(features1['bbox_area'], features2['bbox_area']) + 1e-7)
            
            similarities['shape'] = (ar_sim + area_ratio) / 2
        
        # Weighted combination
        weights = {
            'color_hist_h': 0.15,
            'color_hist_s': 0.15,
            'color_hist_v': 0.10,
            'jersey_color': 0.20,
            'shorts_color': 0.15,
            'hog': 0.15,
            'shape': 0.10
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in similarities:
                total_similarity += similarities[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_similarity / total_weight
        else:
            return 0.0
