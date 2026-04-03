#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from seucm_rectify import build_remap, load_anchor_params, load_seucm_params


def render_video(input_video: Path, calibration_json: Path, params_json: Path, output_video: Path) -> None:
    camera_params = load_seucm_params(calibration_json)
    anchor_params = load_anchor_params(params_json)

    rotate_anchor_x = float(anchor_params["rotate_anchor_x"])
    rotate_anchor_y = float(anchor_params["rotate_anchor_y"])
    shift_x = float(anchor_params["shift_x"])
    shift_y = float(anchor_params["shift_y"])
    roll_degrees = float(anchor_params.get("roll_degrees", 0.0))
    focal_scale = float(anchor_params["focal_scale"])
    output_width = int(anchor_params.get("output_width", 640))
    output_height = int(anchor_params.get("output_height", 480))

    map_x, map_y, valid = build_remap(
        camera_params,
        focal_scale,
        rotate_anchor_x,
        rotate_anchor_y,
        shift_x,
        shift_y,
        roll_degrees,
        output_width,
        output_height,
    )

    print(
        "Using rotate ({:.1f}, {:.1f}), shift ({:.1f}, {:.1f}), roll {:.1f}, focal scale {:.6f}, output {}x{}, valid ratio {:.6f}".format(
            rotate_anchor_x,
            rotate_anchor_y,
            shift_x,
            shift_y,
            roll_degrees,
            focal_scale,
            output_width,
            output_height,
            float(valid.mean()),
        )
    )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {output_video}")

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            writer.write(output)

            frame_index += 1
            if frame_index % 30 == 0 or frame_index == frame_count:
                print(f"Processed {frame_index}/{frame_count} frames")
    finally:
        cap.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an undistorted video from a saved anchor parameter JSON.")
    parser.add_argument("input_video", type=Path, help="Path to the input video")
    parser.add_argument("calibration_json", type=Path, help="Path to the calibration JSON file")
    parser.add_argument("params_json", type=Path, help="Path to the saved anchor parameter JSON")
    parser.add_argument("output_video", type=Path, help="Path to the output video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video(args.input_video, args.calibration_json, args.params_json, args.output_video)


if __name__ == "__main__":
    main()
