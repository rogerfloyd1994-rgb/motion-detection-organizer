#!/usr/bin/env python3
"""
Motion Detection Video Organizer
Detects motion in videos and categorizes by people, pets, or other motion
with timestamping and symlink organization for easy access
"""

import cv2
import os
import glob
import logging
from pathlib import Path
from datetime import datetime
import json
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('motion_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'VIDEO_SEARCH_DIR': '/mnt/sdcard',  # Android tablet SD card
    'MOTION_OUTPUT_DIR': 'motion_videos',
    'SAMPLE_INTERVAL': 10,  # Check every 10th frame
    'MIN_MOTION_AREA': 500,
    'OBJECT_MIN_AREA': 1000,
    'OBJECT_MAX_AREA': 50000,
    'ASPECT_RATIO_MIN': 0.3,
    'ASPECT_RATIO_MAX': 1.5,
}

# Output directory structure
OUTPUT_DIRS = {
    'people': os.path.join(CONFIG['MOTION_OUTPUT_DIR'], datetime.now().strftime('%Y-%m-%d'), 'people'),
    'pets': os.path.join(CONFIG['MOTION_OUTPUT_DIR'], datetime.now().strftime('%Y-%m-%d'), 'pets'),
    'other_motion': os.path.join(CONFIG['MOTION_OUTPUT_DIR'], datetime.now().strftime('%Y-%m-%d'), 'other_motion'),
    'logs': os.path.join(CONFIG['MOTION_OUTPUT_DIR'], 'logs'),
}

class MotionDetector:
    """Main motion detection and categorization class"""
    
    def __init__(self):
        self.setup_directories()
        self.stats = {
            'total_videos': 0,
            'people_videos': 0,
            'pet_videos': 0,
            'other_motion_videos': 0,
            'no_motion_videos': 0,
            'errors': 0
        }
    
    def setup_directories(self):
        """Create necessary output directories"""
        for dir_path in OUTPUT_DIRS.values():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Directory ready: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
    
    def detect_motion(self, frame, background_subtractor):
        """Detect motion in frame using background subtraction"""
        fg_mask = background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = [c for c in contours if cv2.contourArea(c) > CONFIG['MIN_MOTION_AREA']]
        return len(significant_contours) > 0, fg_mask
    
    def detect_objects(self, frame):
        """
        Detect potential people/pets using edge detection and contour analysis
        Returns tuple: (has_objects, object_type_confidence)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Check if object matches person/pet size range
            if (CONFIG['OBJECT_MIN_AREA'] < area < CONFIG['OBJECT_MAX_AREA'] and
                CONFIG['ASPECT_RATIO_MIN'] < aspect_ratio < CONFIG['ASPECT_RATIO_MAX']):
                
                detected_objects.append({
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'height': h,
                    'width': w
                })
        
        return len(detected_objects) > 0, detected_objects
    
    def classify_object_type(self, objects):
        """
        Classify detected objects as people or pets based on size and aspect ratio
        Returns: 'people', 'pets', or None
        """
        if not objects:
            return None
        
        avg_area = sum(obj['area'] for obj in objects) / len(objects)
        avg_height = sum(obj['height'] for obj in objects) / len(objects)
        
        # Heuristic: Larger vertical objects (avg_height > avg_width) likely people
        # Smaller or squatter objects likely pets
        avg_aspect = sum(obj['aspect_ratio'] for obj in objects) / len(objects)
        
        if avg_area > 15000 and avg_aspect < 0.8:
            return 'people'
        elif avg_area < 15000 and avg_height < 200:
            return 'pets'
        else:
            # Default to people if dimensions suggest upright posture
            return 'people' if avg_height > avg_area / 50 else 'pets'
    
    def create_symlink(self, source, destination):
        """Create symlink to source file in destination directory"""
        try:
            link_path = os.path.join(destination, os.path.basename(source))
            
            # Remove existing link if present
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            
            # Create symlink (works on Windows with Python 3.8+, Linux, macOS)
            if platform.system() == 'Windows':
                # On Windows, use mklink via command line
                os.system(f'mklink "{link_path}" "{source}"')
            else:
                # Unix-like systems
                os.symlink(source, link_path)
            
            return True
        except Exception as e:
            logger.warning(f"Could not create symlink: {str(e)}")
            return False
    
    def process_video(self, video_path):
        """Process single video file"""
        logger.info(f"Processing: {video_path}")
        self.stats['total_videos'] += 1
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                self.stats['errors'] += 1
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            
            has_motion = False
            object_type = None
            detected_frames = []
            processed_frames = 0
            
            frame_index = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_index += 1
                
                # Sample frames to reduce processing load
                if frame_index % CONFIG['SAMPLE_INTERVAL'] != 0:
                    continue
                
                processed_frames += 1
                
                # Detect motion
                motion_detected, _ = detect_motion(frame, background_subtractor)
                
                if motion_detected:
                    has_motion = True
                    
                    # Check for objects
                    objects_detected, objects = self.detect_objects(frame)
                    
                    if objects_detected:
                        classified_type = self.classify_object_type(objects)
                        if classified_type:
                            object_type = classified_type
                            timestamp = frame_index / fps if fps > 0 else 0
                            detected_frames.append({
                                'frame': frame_index,
                                'timestamp': timestamp,
                                'type': classified_type
                            })
            
            cap.release()
            
            # Categorize and save
            self.categorize_and_save(video_path, has_motion, object_type, detected_frames, duration_seconds)
        
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            self.stats['errors'] += 1
    
    def categorize_and_save(self, video_path, has_motion, object_type, detected_frames, duration):
        """Categorize video and create symlink + metadata"""
        if not has_motion:
            self.stats['no_motion_videos'] += 1
            logger.info(f"  → No motion detected")
            return
        
        # Determine destination
        if object_type == 'people':
            destination = OUTPUT_DIRS['people']
            category = 'people'
            self.stats['people_videos'] += 1
        elif object_type == 'pets':
            destination = OUTPUT_DIRS['pets']
            category = 'pets'
            self.stats['pet_videos'] += 1
        else:
            destination = OUTPUT_DIRS['other_motion']
            category = 'other_motion'
            self.stats['other_motion_videos'] += 1
        
        # Create symlink
        self.create_symlink(video_path, destination)
        
        # Create metadata file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.basename(video_path)
        metadata_filename = f"{timestamp}_{video_filename}.json"
        metadata_path = os.path.join(destination, metadata_filename)
        
        metadata = {
            'source_path': video_path,
            'category': category,
            'processed_timestamp': datetime.now().isoformat(),
            'video_duration_seconds': duration,
            'detected_frames': detected_frames,
            'detection_count': len(detected_frames)
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"  → Categorized as: {category} ({len(detected_frames)} detections)")
        except Exception as e:
            logger.error(f"Failed to write metadata: {str(e)}")
    
    def scan_and_process(self, search_dir):
        """Scan directory for videos and process them"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
        
        logger.info(f"Found {len(video_files)} video files in {search_dir}")
        
        if not video_files:
            logger.warning("No video files found!")
            return
        
        for video_path in video_files:
            self.process_video(video_path)
        
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        logger.info("\n" + "="*60)
        logger.info("MOTION DETECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos processed: {self.stats['total_videos']}")
        logger.info(f"People detected: {self.stats['people_videos']}")
        logger.info(f"Pets detected: {self.stats['pet_videos']}")
        logger.info(f"Other motion: {self.stats['other_motion_videos']}")
        logger.info(f"No motion: {self.stats['no_motion_videos']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*60)
        logger.info(f"Output directory: {CONFIG['MOTION_OUTPUT_DIR']}")
        logger.info("="*60 + "\n")

def main():
    """Main entry point"""
    logger.info("Motion Detection Video Organizer Started")
    logger.info(f"Search directory: {CONFIG['VIDEO_SEARCH_DIR']}")
    
    detector = MotionDetector()
    detector.scan_and_process(CONFIG['VIDEO_SEARCH_DIR'])
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
