# Motion Detection Video Organizer

A Python project for scanning video files or camera feeds, detecting motion, classifying likely activity (people, pets, or other motion), and organizing results into timestamped output folders.

This project is designed for Android tablet and local video workflows, with optional motion clipping, configurable ROI zones, and a YOLO-compatible detector path.

## Features

- Scan directories recursively for video files
- Process live camera or video sources
- Detect motion using background subtraction
- Classify detected motion into:
  - people
  - pets
  - other motion
- Organize results into date-stamped output folders
- Support configurable monitoring zones and exclusion polygons
- Save metadata JSON files with detection details
- Save motion event clips when enabled
- Optional YOLOv8-compatible detection pipeline

## Repository layout

- `motion_detector.py` — main detection and processing logic
- `app.py` — small entry-point wrapper for local or Docker execution
- `sample_config.json` — example configuration file
- `Dockerfile` — container build for running the app in a controlled environment
- `requirements.txt` — Python dependencies

## Quick start

### Local environment

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the detector against a directory of videos:

   ```bash
   python motion_detector.py --search-dir /path/to/videos --output-dir motion_videos
   ```

4. Or run a configured camera source:

   ```bash
   python motion_detector.py --camera 0 --detector heuristic
   ```

## Example configuration

The project includes a sample config file:

```bash
python motion_detector.py --config sample_config.json
```

Example values in `sample_config.json` include:

```json
{
  "VIDEO_SEARCH_DIR": "/mnt/sdcard",
  "MOTION_OUTPUT_DIR": "motion_videos",
  "SAMPLE_INTERVAL": 5,
  "DETECTOR": "yolov8",
  "RECORD_CLIPS": true,
  "CLIP_DIR": "motion_clips",
  "ZONES": [[ [100,100], [500,100], [500,400], [100,400] ]],
  "EXCLUSIONS": [[ [220,220], [360,220], [360,360], [220,360] ]]
}
```

## Configuration options

Common options:

- `--search-dir`: directory containing videos to scan
- `--output-dir`: base directory for organized motion results
- `--camera`: camera index or video path to process directly
- `--sample-interval`: process every Nth frame
- `--min-motion-area`: minimum area required to count motion
- `--zone`: polygon ROI in `x,y:x,y:...` format
- `--exclude`: polygon exclusion in `x,y:x,y:...` format
- `--detector`: `heuristic`, `hog`, `yolov8`, or `yolo`
- `--config`: path to a JSON configuration file
- `--record-clips`: save short clips when motion is detected
- `--clip-dir`: directory for saved clips
- `--model-path`: optional YOLO model path
- `--confidence`: minimum detection confidence for YOLO-based detection

## Docker usage

Build the container:

```bash
docker build -t motion-detection-organizer .
```

Run it:

```bash
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  motion-detection-organizer --camera 0
```

For file-based scanning, mount a local folder and point to it:

```bash
docker run --rm -it \
  -v /path/to/videos:/data \
  motion-detection-organizer --search-dir /data --output-dir /data/output
```

## Output structure

When motion is detected, the app creates output folders like:

```text
motion_videos/
  2026-10-25/
    people/
    pets/
    other_motion/
    logs/
```

Each categorized result includes a metadata JSON file with timestamps, detected frames, categories, and config information.

## Notes

- The heuristic detector is the default and is the safest option for most setups.
- YOLO-based detection requires the `ultralytics` dependency and a valid model path or default model download behavior.
- Some systems may require additional OS libraries for OpenCV and camera access.

## Testing

Run the minimal test suite with:

```bash
pytest -q
```
