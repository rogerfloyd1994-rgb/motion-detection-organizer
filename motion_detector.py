#!/usr/bin/env python3
"""
Motion Detection Video Organizer
Detects motion in videos and categorizes by people, pets, or other motion
with timestamping and symlink organization for easy access.

Additional features:
- Scan directories for videos or process live camera feeds
- Select one or more cameras using --camera
- Define monitoring zones and exclusion zones
- Support JSON config files
- Optional HOG person detection mode
- Optional clip recording on detected events
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# Default configuration
DEFAULT_CONFIG = {
    "VIDEO_SEARCH_DIR": "/mnt/sdcard",
    "MOTION_OUTPUT_DIR": "motion_videos",
    "SAMPLE_INTERVAL": 10,
    "MIN_MOTION_AREA": 500,
    "OBJECT_MIN_AREA": 1000,
    "OBJECT_MAX_AREA": 50000,
    "ASPECT_RATIO_MIN": 0.3,
    "ASPECT_RATIO_MAX": 1.5,
    "RESIZE_WIDTH": 640,
    "MAX_LOG_BYTES": 5 * 1024 * 1024,
    "BACKUP_COUNT": 3,
    "VIDEO_EXTENSIONS": ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv"],
    "CAMERAS": [],
    "ZONES": [],
    "EXCLUSIONS": [],
    "DETECTOR": "heuristic",
    "RECORD_CLIPS": False,
    "CLIP_DIR": "motion_clips",
    "HOG_SCALE": 1.05,
    "HOG_STRIDE": [4, 4],
    "HOG_PADDING": [8, 8],
    "HOG_THRESHOLD": 0.0,
}


def parse_camera_source(value: str):
    """Convert camera string into int or file/device path."""
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_polygon(value: str):
    """
    Parse a polygon string of the form:
      "100,100:300,100:300,300:100,300"
      "100,100 300,100 300,300 100,300"
      "100,100;300,100;300,300;100,300"
    """
    if value is None:
        return []

    cleaned = value.strip()
    if not cleaned:
        return []

    parts = [p for p in re.split(r"[:,;\s]+", cleaned) if p]
    if len(parts) < 4 or len(parts) % 2 != 0:
        return []

    points = []
    for i in range(0, len(parts), 2):
        try:
            x = float(parts[i])
            y = float(parts[i + 1])
            points.append((int(x), int(y)))
        except (TypeError, ValueError):
            continue

    if len(points) < 3:
        return []
    return points


def safe_find_contours(mask):
    """Handle OpenCV contour API differences."""
    result = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return contours, hierarchy


@dataclass
class CameraConfig:
    source: Union[int, str]
    name: str
    zones: List[List[Tuple[int, int]]] = field(default_factory=list)
    exclusions: List[List[Tuple[int, int]]] = field(default_factory=list)


class MotionDetector:
    """Main motion detection and categorization class."""

    def __init__(self, config=None, cameras=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.cameras = cameras or self._build_default_cameras()
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.output_base = Path(self.config["MOTION_OUTPUT_DIR"]).resolve()
        self.output_dirs = {
            "people": self.output_base / self.date_str / "people",
            "pets": self.output_base / self.date_str / "pets",
            "other_motion": self.output_base / self.date_str / "other_motion",
            "logs": self.output_base / "logs",
        }
        self.clip_root = Path(self.config.get("CLIP_DIR", "motion_clips")).resolve()
        self.clip_root.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        self.setup_directories()
        self.stats = {
            "total_videos": 0,
            "people_videos": 0,
            "pet_videos": 0,
            "other_motion_videos": 0,
            "no_motion_videos": 0,
            "errors": 0,
        }

    def _normalize_zones(self, zones):
        normalized = []
        for zone in zones:
            if isinstance(zone, str):
                parsed = parse_polygon(zone)
                if parsed:
                    normalized.append(parsed)
            elif isinstance(zone, (list, tuple)) and zone and isinstance(zone[0], (list, tuple)):
                normalized.append([(int(x), int(y)) for x, y in zone])
            elif isinstance(zone, (list, tuple)) and len(zone) >= 2 and isinstance(zone[0], (int, float)):
                transformed = []
                for i in range(0, len(zone), 2):
                    if i + 1 < len(zone):
                        transformed.append((int(zone[i]), int(zone[i + 1])))
                if len(transformed) >= 3:
                    normalized.append(transformed)
        return normalized

    def _build_default_cameras(self):
        configured = self.config.get("CAMERAS", [])
        if not configured:
            return [
                CameraConfig(
                    source=self.config["VIDEO_SEARCH_DIR"],
                    name="search_dir",
                    zones=self._normalize_zones(self.config.get("ZONES", [])),
                    exclusions=self._normalize_zones(self.config.get("EXCLUSIONS", [])),
                )
            ]

        result = []
        for idx, camera in enumerate(configured):
            if isinstance(camera, CameraConfig):
                result.append(camera)
            else:
                result.append(
                    CameraConfig(
                        source=camera.get("source", self.config["VIDEO_SEARCH_DIR"]),
                        name=camera.get("name", f"camera_{idx}"),
                        zones=self._normalize_zones(camera.get("zones", self.config.get("ZONES", []))),
                        exclusions=self._normalize_zones(camera.get("exclusions", self.config.get("EXCLUSIONS", []))),
                    )
                )
        return result

    def _setup_logging(self):
        log_dir = self.output_dirs["logs"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "motion_detector.log"

        logger = logging.getLogger("motion_detector")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=self.config["MAX_LOG_BYTES"],
            backupCount=self.config["BACKUP_COUNT"],
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def setup_directories(self):
        for dir_path in self.output_dirs.values():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Directory ready: {dir_path}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {dir_path}: {str(e)}")

    def resize_frame(self, frame):
        resize_w = self.config.get("RESIZE_WIDTH", 0)
        if resize_w and frame.shape[1] > resize_w:
            scale = resize_w / float(frame.shape[1])
            new_h = max(1, int(frame.shape[0] * scale))
            return cv2.resize(frame, (resize_w, new_h))
        return frame

    def build_roi_mask(self, frame_shape, zones=None, exclusions=None):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not zones:
            mask[:] = 255
            if exclusions:
                exclude_mask = np.zeros((h, w), dtype=np.uint8)
                for poly in exclusions:
                    if len(poly) >= 3:
                        pts = np.array(poly, dtype=np.int32)
                        cv2.fillPoly(exclude_mask, [pts], 255)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))
            return mask

        for poly in zones:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        if exclusions:
            exclude_mask = np.zeros((h, w), dtype=np.uint8)
            for poly in exclusions:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(exclude_mask, [pts], 255)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

        return mask

    def contour_in_roi(self, contour, zones=None, exclusions=None):
        if not zones:
            return True

        centroid = cv2.moments(contour)
        if centroid["m00"] == 0:
            return False

        cx = centroid["m10"] / centroid["m00"]
        cy = centroid["m01"] / centroid["m00"]

        point = (float(cx), float(cy))
        for poly in zones:
            pts = np.array(poly, dtype=np.int32)
            result = cv2.pointPolygonTest(pts, point, False)
            if result >= 0:
                if exclusions:
                    for ex_poly in exclusions:
                        ex_pts = np.array(ex_poly, dtype=np.int32)
                        ex_result = cv2.pointPolygonTest(ex_pts, point, False)
                        if ex_result >= 0:
                            return False
                return True

        return False

    def detect_motion(self, frame, background_subtractor, zones=None, exclusions=None):
        frame = self.resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = background_subtractor.apply(gray)

        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=2)

        roi_mask = self.build_roi_mask(cleaned.shape, zones=zones, exclusions=exclusions)
        cleaned = cv2.bitwise_and(cleaned, roi_mask)

        contours, _ = safe_find_contours(cleaned)
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.config["MIN_MOTION_AREA"]]
        return len(significant_contours) > 0, cleaned

    def _filter_roi(self, boxes, zones=None, exclusions=None):
        selected = []
        for box in boxes:
            x, y, w, h = box
            cx = x + (w / 2)
            cy = y + (h / 2)
            point = (float(cx), float(cy))

            inside_zone = True
            if zones:
                inside_zone = False
                for poly in zones:
                    pts = np.array(poly, dtype=np.int32)
                    result = cv2.pointPolygonTest(pts, point, False)
                    if result >= 0:
                        inside_zone = True
                        break

            if not inside_zone:
                continue

            if exclusions:
                for poly in exclusions:
                    pts = np.array(poly, dtype=np.int32)
                    result = cv2.pointPolygonTest(pts, point, False)
                    if result >= 0:
                        inside_zone = False
                        break
            if inside_zone:
                selected.append({
                    "area": int(w * h),
                    "aspect_ratio": float(w) / float(h) if h > 0 else 0.0,
                    "height": int(h),
                    "width": int(w),
                    "bbox": [int(x), int(y), int(w), int(h)],
                })
        return selected

    def detect_hog_objects(self, frame, zones=None, exclusions=None):
        frame = self.resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        boxes, _ = hog.detectMultiScale(
            gray,
            winStride=(int(self.config.get("HOG_STRIDE", [4, 4])[0]), int(self.config.get("HOG_STRIDE", [4, 4])[1])),
            padding=(int(self.config.get("HOG_PADDING", [8, 8])[0]), int(self.config.get("HOG_PADDING", [8, 8])[1])),
            scale=float(self.config.get("HOG_SCALE", 1.05)),
            hitThreshold=float(self.config.get("HOG_THRESHOLD", 0.0)),
        )

        if boxes is None or len(boxes) == 0:
            return False, []

        filtered = self._filter_roi([tuple(b) for b in boxes], zones, exclusions)
        return len(filtered) > 0, filtered

    def detect_objects(self, frame, zones=None, exclusions=None):
        detector_type = str(self.config.get("DETECTOR", "heuristic")).lower()
        if detector_type == "hog":
            return self.detect_hog_objects(frame, zones, exclusions)

        frame = self.resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        contours, _ = safe_find_contours(edges)
        detected_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["OBJECT_MIN_AREA"] or area > self.config["OBJECT_MAX_AREA"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0:
                continue

            aspect_ratio = float(w) / float(h)
            if not (self.config["ASPECT_RATIO_MIN"] < aspect_ratio < self.config["ASPECT_RATIO_MAX"]):
                continue

            if not self.contour_in_roi(contour, zones=zones, exclusions=exclusions):
                continue

            detected_objects.append({
                "area": area,
                "aspect_ratio": aspect_ratio,
                "height": h,
                "width": w,
                "bbox": [int(x), int(y), int(w), int(h)],
            })

        return len(detected_objects) > 0, detected_objects

    def classify_object_type(self, objects):
        if not objects:
            return None

        avg_area = sum(obj["area"] for obj in objects) / len(objects)
        avg_height = sum(obj["height"] for obj in objects) / len(objects)
        avg_aspect = sum(obj["aspect_ratio"] for obj in objects) / len(objects)

        if avg_area > 20000 and avg_aspect < 0.8:
            return "people"
        elif avg_area < 8000 or avg_height < 200:
            return "pets"
        else:
            return "people" if avg_height > (avg_area ** 0.5) else "pets"

    def create_symlink(self, source, destination):
        """Create symlink with hardlink or copy fallback."""
        source_path = Path(source)
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)
        link_path = dest_dir / source_path.name

        try:
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()

            try:
                link_path.symlink_to(source_path.resolve())
                self.logger.debug(f"Created symlink: {link_path} -> {source_path}")
                return True
            except (OSError, NotImplementedError):
                try:
                    os.link(str(source_path), str(link_path))
                    self.logger.debug(f"Created hard link: {link_path} -> {source_path}")
                    return True
                except OSError:
                    shutil.copy2(str(source_path), str(link_path))
                    self.logger.debug(f"Copied file: {source_path} -> {link_path}")
                    return True
        except Exception as e:
            self.logger.warning(f"Could not create link/copy for {source}: {str(e)}")
            return False

    def save_event_clip(self, frames, label):
        if not frames:
            return

        if not self.config.get("RECORD_CLIPS", False):
            return

        clip_dir = Path(self.config.get("CLIP_DIR", "motion_clips")).resolve()
        clip_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = clip_dir / f"{label}_{timestamp}.mp4"

        if not frames:
            return

        height, width = frames[0].shape[:2]
        fps = 10.0
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            self.logger.warning(f"Could not create clip writer for {clip_path}")
            return

        for frame in frames:
            writer.write(frame)
        writer.release()
        self.logger.info(f"Saved clip: {clip_path}")

    def process_camera(self, camera: CameraConfig):
        self.logger.info(f"Opening camera/source: {camera.name} -> {camera.source}")
        cap = cv2.VideoCapture(camera.source)
        if not cap.isOpened():
            self.logger.error(f"Could not open camera/source: {camera.name} ({camera.source})")
            self.stats["errors"] += 1
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = frame_count / fps if fps > 0 else 0.0

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        event_frames = []
        has_motion = False
        object_type = None
        detected_frames = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % self.config["SAMPLE_INTERVAL"] != 0:
                continue

            motion_detected, _ = self.detect_motion(frame, bg_sub, camera.zones, camera.exclusions)
            if motion_detected:
                has_motion = True
                if self.config.get("RECORD_CLIPS", False):
                    event_frames.append(frame.copy())
                    if len(event_frames) > 30:
                        event_frames.pop(0)

                objects_detected, objects = self.detect_objects(frame, camera.zones, camera.exclusions)
                if objects_detected:
                    classified_type = self.classify_object_type(objects)
                    if classified_type:
                        object_type = classified_type
                        ts = frame_index / fps if fps > 0 else 0.0
                        detected_frames.append({
                            "frame": frame_index,
                            "timestamp": ts,
                            "type": classified_type,
                        })

        cap.release()
        self.categorize_and_save(
            source_path=str(camera.source),
            label=camera.name,
            has_motion=has_motion,
            object_type=object_type,
            detected_frames=detected_frames,
            duration=duration_seconds,
            event_frames=event_frames,
        )

    def process_video(self, video_path):
        self.logger.info(f"Processing: {video_path}")
        self.stats["total_videos"] += 1

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                self.stats["errors"] += 1
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_seconds = frame_count / fps if fps > 0 else 0.0

            bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
            has_motion = False
            object_type = None
            detected_frames = []
            event_frames = []
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                if frame_index % self.config["SAMPLE_INTERVAL"] != 0:
                    continue

                motion_detected, _ = self.detect_motion(
                    frame,
                    bg_sub,
                    self.config.get("ZONES", []),
                    self.config.get("EXCLUSIONS", []),
                )
                if motion_detected:
                    has_motion = True
                    if self.config.get("RECORD_CLIPS", False):
                        event_frames.append(frame.copy())
                        if len(event_frames) > 30:
                            event_frames.pop(0)
                    objects_detected, objects = self.detect_objects(
                        frame,
                        self.config.get("ZONES", []),
                        self.config.get("EXCLUSIONS", []),
                    )
                    if objects_detected:
                        classified_type = self.classify_object_type(objects)
                        if classified_type:
                            object_type = classified_type
                            ts = frame_index / fps if fps > 0 else 0.0
                            detected_frames.append({
                                "frame": frame_index,
                                "timestamp": ts,
                                "type": classified_type,
                            })

            self.categorize_and_save(
                source_path=video_path,
                label=Path(video_path).name,
                has_motion=has_motion,
                object_type=object_type,
                detected_frames=detected_frames,
                duration=duration_seconds,
                event_frames=event_frames,
            )

        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {str(e)}")
            self.stats["errors"] += 1
        finally:
            if cap is not None:
                cap.release()

    def categorize_and_save(self, source_path, label, has_motion, object_type, detected_frames, duration, event_frames=None):
        if not has_motion:
            self.stats["no_motion_videos"] += 1
            self.logger.info(f"  → No motion detected for {label}")
            return

        if object_type == "people":
            destination = self.output_dirs["people"]
            category = "people"
            self.stats["people_videos"] += 1
        elif object_type == "pets":
            destination = self.output_dirs["pets"]
            category = "pets"
            self.stats["pet_videos"] += 1
        else:
            destination = self.output_dirs["other_motion"]
            category = "other_motion"
            self.stats["other_motion_videos"] += 1

        self.create_symlink(source_path, str(destination))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = Path(source_path).name
        metadata_filename = f"{timestamp}_{uuid.uuid4().hex}_{video_filename}.json"
        metadata_path = Path(destination) / metadata_filename

        metadata = {
            "source_path": str(source_path),
            "label": label,
            "category": category,
            "processed_timestamp": datetime.now().isoformat(),
            "video_duration_seconds": duration,
            "detected_frames": detected_frames,
            "detection_count": len(detected_frames),
            "zones": self.config.get("ZONES", []),
            "exclusions": self.config.get("EXCLUSIONS", []),
            "detector": self.config.get("DETECTOR", "heuristic"),
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"  → Categorized as: {category} ({len(detected_frames)} detections)")
        except Exception as e:
            self.logger.error(f"Failed to write metadata: {str(e)}")

        if self.config.get("RECORD_CLIPS", False) and event_frames:
            self.save_event_clip(event_frames, f"{category}_{uuid.uuid4().hex[:8]}")

    def scan_and_process(self, search_dir):
        video_files = set()
        for ext in self.config["VIDEO_EXTENSIONS"]:
            for match in glob.glob(os.path.join(search_dir, "**", ext), recursive=True):
                if os.path.isfile(match):
                    video_files.add(os.path.abspath(match))

        video_files = sorted(video_files)
        self.logger.info(f"Found {len(video_files)} video files in {search_dir}")

        if not video_files:
            self.logger.warning("No video files found!")
            return

        for video_path in video_files:
            self.process_video(video_path)

        self.print_summary()

    def print_summary(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MOTION DETECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total videos processed: {self.stats['total_videos']}")
        self.logger.info(f"People detected: {self.stats['people_videos']}")
        self.logger.info(f"Pets detected: {self.stats['pet_videos']}")
        self.logger.info(f"Other motion: {self.stats['other_motion_videos']}")
        self.logger.info(f"No motion: {self.stats['no_motion_videos']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_base}")
        self.logger.info("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Motion Detection Video Organizer")
    parser.add_argument("--search-dir", default=DEFAULT_CONFIG["VIDEO_SEARCH_DIR"], help="Directory to scan for video files")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["MOTION_OUTPUT_DIR"], help="Base output directory")
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_CONFIG["SAMPLE_INTERVAL"], help="Check every Nth frame")
    parser.add_argument("--resize-width", type=int, default=DEFAULT_CONFIG["RESIZE_WIDTH"], help="Resize width for processing (0 = disabled)")
    parser.add_argument("--min-motion-area", type=int, default=DEFAULT_CONFIG["MIN_MOTION_AREA"], help="Minimum contour area to count as motion")
    parser.add_argument("--camera", action="append", default=[], help="Camera or video source to process. Repeat for multiple sources. Examples: --camera 0 --camera /dev/video0 --camera camera.mp4")
    parser.add_argument("--zone", action="append", default=[], help="Polygon monitoring zone in x,y:x,y:... format. Repeat for multiple zones.")
    parser.add_argument("--exclude", action="append", default=[], help="Polygon exclusion in x,y:x,y:... format. Repeat for multiple exclusions.")
    parser.add_argument("--config", default=None, help="JSON configuration file path")
    parser.add_argument("--detector", choices=["heuristic", "hog"], default=DEFAULT_CONFIG["DETECTOR"], help="Detection mode")
    parser.add_argument("--record-clips", action="store_true", default=DEFAULT_CONFIG["RECORD_CLIPS"], help="Save short event clips when motion is detected")
    parser.add_argument("--clip-dir", default=DEFAULT_CONFIG["CLIP_DIR"], help="Directory used for saved event clips")
    return parser.parse_args()


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded


def build_camera_configs(args, extra_config=None):
    config = extra_config or {}

    file_zones = config.get("zones", [])
    file_exclusions = config.get("exclusions", [])
    file_cameras = config.get("cameras", [])

    cli_zone_polygons = [parse_polygon(z) for z in args.zone]
    cli_exclude_polygons = [parse_polygon(z) for z in args.exclude]
    config_zones = [z for z in list(file_zones) + cli_zone_polygons if z]
    config_exclusions = [z for z in list(file_exclusions) + cli_exclude_polygons if z]

    if args.camera:
        cameras = []
        for idx, source in enumerate(args.camera):
            source_value = parse_camera_source(source)
            cameras.append(
                CameraConfig(
                    source=source_value,
                    name=f"camera_{idx}",
                    zones=config_zones,
                    exclusions=config_exclusions,
                )
            )
        return cameras, config_zones, config_exclusions

    if file_cameras:
        cameras = []
        for idx, source in enumerate(file_cameras):
            if isinstance(source, dict):
                source_value = parse_camera_source(str(source.get("source", idx)))
                name = source.get("name", f"camera_{idx}")
                zones = source.get("zones", config_zones)
                exclusions = source.get("exclusions", config_exclusions)
            else:
                source_value = parse_camera_source(str(source))
                name = f"camera_{idx}"
                zones = config_zones
                exclusions = config_exclusions

            zone_list = []
            for item in zones:
                if isinstance(item, str):
                    parsed = parse_polygon(item)
                    if parsed:
                        zone_list.append(parsed)
                elif isinstance(item, list):
                    zone_list.append([(int(x), int(y)) for x, y in item])

            exclusion_list = []
            for item in exclusions:
                if isinstance(item, str):
                    parsed = parse_polygon(item)
                    if parsed:
                        exclusion_list.append(parsed)
                elif isinstance(item, list):
                    exclusion_list.append([(int(x), int(y)) for x, y in item])

            cameras.append(
                CameraConfig(
                    source=source_value,
                    name=name,
                    zones=zone_list,
                    exclusions=exclusion_list,
                )
            )
        return cameras, config_zones, config_exclusions

    cameras = [
        CameraConfig(
            source=args.search_dir,
            name="search_dir",
            zones=config_zones,
            exclusions=config_exclusions,
        )
    ]
    return cameras, config_zones, config_exclusions


def main():
    args = parse_args()

    extra_config = {}
    if args.config:
        extra_config = load_json_config(args.config)

    cameras, zone_list, exclusion_list = build_camera_configs(args, extra_config)

    config = {
        "VIDEO_SEARCH_DIR": args.search_dir,
        "MOTION_OUTPUT_DIR": args.output_dir,
        "SAMPLE_INTERVAL": args.sample_interval,
        "RESIZE_WIDTH": args.resize_width,
        "MIN_MOTION_AREA": args.min_motion_area,
        "ZONES": zone_list,
        "EXCLUSIONS": exclusion_list,
        "CAMERAS": [
            {
                "source": camera.source,
                "name": camera.name,
                "zones": camera.zones,
                "exclusions": camera.exclusions,
            }
            for camera in cameras
        ],
        "DETECTOR": args.detector,
        "RECORD_CLIPS": bool(args.record_clips),
        "CLIP_DIR": args.clip_dir,
    }
    config.update(extra_config)

    detector = MotionDetector(config=config, cameras=cameras)
    detector.logger.info("Motion Detection Video Organizer Started")
    detector.logger.info(f"Search directory: {args.search_dir}")
    detector.logger.info(f"Zones configured: {zone_list}")
    detector.logger.info(f"Exclusions configured: {exclusion_list}")
    detector.logger.info(f"Detection mode: {args.detector}")

    if args.camera or args.config:
        for camera in cameras:
            detector.process_camera(camera)
    else:
        detector.scan_and_process(args.search_dir)

    detector.logger.info("Processing complete!")


if __name__ == "__main__":
    main()
