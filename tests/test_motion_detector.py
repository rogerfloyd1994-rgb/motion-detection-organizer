import json
from pathlib import Path

import pytest

from motion_detector import MotionDetector, parse_polygon


def test_parse_polygon_parses_expected_coordinates():
    coords = parse_polygon("100,100:300,100:300,300:100,300")
    assert coords == [(100, 100), (300, 100), (300, 300), (100, 300)]


def test_motion_detector_creates_expected_directories(tmp_path):
    output_dir = tmp_path / "motion_videos"
    detector = MotionDetector(
        config={
            "VIDEO_SEARCH_DIR": str(tmp_path),
            "MOTION_OUTPUT_DIR": str(output_dir),
            "CAMERAS": [],
            "DETECTOR": "heuristic",
        }
    )

    assert output_dir.exists()
    assert (output_dir / detector.date_str / "people").exists()
    assert (output_dir / detector.date_str / "pets").exists()
    assert (output_dir / detector.date_str / "other_motion").exists()
    assert (output_dir / "logs").exists()


def test_validate_runtime_config_rejects_invalid_detector(tmp_path):
    bad_config = {
        "VIDEO_SEARCH_DIR": str(tmp_path),
        "MOTION_OUTPUT_DIR": str(tmp_path / "motion_videos"),
        "DETECTOR": "unsupported",
        "CAMERAS": [],
    }

    with pytest.raises(ValueError, match="Unsupported detector"):
        MotionDetector(config=bad_config)


def test_load_json_config_reads_valid_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"VIDEO_SEARCH_DIR": str(tmp_path), "DETECTOR": "heuristic"}), encoding="utf-8")

    from motion_detector import load_json_config

    loaded = load_json_config(str(config_path))
    assert loaded["VIDEO_SEARCH_DIR"] == str(tmp_path)
    assert loaded["DETECTOR"] == "heuristic"
