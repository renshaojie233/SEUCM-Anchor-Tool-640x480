#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk

from seucm_rectify import (
    extract_view_outline,
    find_full_view_scale,
    load_anchor_params,
    load_seucm_params,
    remap_frame,
    save_anchor_params,
)


class AnchorGui:
    def __init__(self, root: tk.Tk, video_path: Path, calibration_path: Path, params_path: Path) -> None:
        self.root = root
        self.video_path = video_path
        self.calibration_path = calibration_path
        self.params_path = params_path
        self.output_width = 640
        self.output_height = 480
        self.source_preview_size = 480
        self.after_id = None
        self.source_photo = None
        self.output_photo = None
        self.current_outline = []
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open input video")
        self.frame_count = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.current_frame_index = -1

        self.camera_params = load_seucm_params(calibration_path)
        self.frame_bgr = None
        self.frame_rgb = None
        self._load_frame(0)
        self.source_height, self.source_width = self.frame_rgb.shape[:2]

        defaults = {
            "rotate_anchor_x": (self.source_width - 1) / 2.0,
            "rotate_anchor_y": (self.source_height - 1) / 2.0,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "roll_degrees": 0.0,
            "focal_scale": 1.0,
        }
        if params_path.exists():
            defaults.update(load_anchor_params(params_path))

        self.rotate_anchor_x_var = tk.DoubleVar(value=float(defaults["rotate_anchor_x"]))
        self.rotate_anchor_y_var = tk.DoubleVar(value=float(defaults["rotate_anchor_y"]))
        self.shift_x_var = tk.DoubleVar(value=float(defaults["shift_x"]))
        self.shift_y_var = tk.DoubleVar(value=float(defaults["shift_y"]))
        self.roll_degrees_var = tk.DoubleVar(value=float(defaults["roll_degrees"]))
        self.focal_scale_var = tk.DoubleVar(value=float(defaults["focal_scale"]))
        self.frame_index_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Click source image or use sliders, then save parameters.")
        self.stats_var = tk.StringVar(value="")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._update_preview()

    def _load_frame(self, frame_index: int) -> None:
        frame_index = min(max(int(frame_index), 0), self.frame_count - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame {} from input video".format(frame_index))
        self.current_frame_index = frame_index
        self.frame_bgr = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _build_ui(self) -> None:
        self.root.title("SEUCM Anchor Tool 640x480")

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        preview_frame = tk.Frame(main)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(preview_frame)
        left_frame.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(left_frame, text="Source 1280x1280").pack()
        self.source_canvas = tk.Canvas(
            left_frame,
            width=self.source_preview_size,
            height=self.source_preview_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.source_canvas.pack()
        self.source_canvas.bind("<Button-1>", self._on_source_click)

        right_frame = tk.Frame(preview_frame)
        right_frame.pack(side=tk.LEFT)
        tk.Label(right_frame, text="Output Preview 640x480").pack()
        self.output_label = tk.Label(
            right_frame,
            width=self.output_width,
            height=self.output_height,
            bd=1,
            relief=tk.SOLID,
        )
        self.output_label.pack()

        controls = tk.Frame(main)
        controls.pack(fill=tk.X, pady=(12, 0))

        tk.Label(controls, text="Rotate Anchor X").grid(row=0, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0,
            to=self.source_width - 1,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.rotate_anchor_x_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=0, column=1, sticky="we")

        tk.Label(controls, text="Rotate Anchor Y").grid(row=1, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0,
            to=self.source_height - 1,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.rotate_anchor_y_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=1, column=1, sticky="we")

        tk.Label(controls, text="Shift X").grid(row=2, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=-self.output_width,
            to=self.output_width,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.shift_x_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=2, column=1, sticky="we")

        tk.Label(controls, text="Shift Y").grid(row=3, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=-self.output_height,
            to=self.output_height,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.shift_y_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=3, column=1, sticky="we")

        tk.Label(controls, text="Roll Degrees").grid(row=4, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            variable=self.roll_degrees_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=4, column=1, sticky="we")

        tk.Label(controls, text="Focal Scale").grid(row=5, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0.05,
            to=4.0,
            orient=tk.HORIZONTAL,
            resolution=0.001,
            variable=self.focal_scale_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=5, column=1, sticky="we")

        tk.Label(controls, text="Frame").grid(row=6, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0,
            to=self.frame_count - 1,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.frame_index_var,
            length=420,
            command=self._on_control_change,
        ).grid(row=6, column=1, sticky="we")

        button_frame = tk.Frame(controls)
        button_frame.grid(row=0, column=2, rowspan=7, padx=(16, 0), sticky="ns")
        tk.Button(button_frame, text="Auto Fit", width=14, command=self._auto_fit).pack(pady=(0, 8))
        tk.Button(button_frame, text="Confirm / Save", width=14, command=self._save_params).pack()

        controls.grid_columnconfigure(1, weight=1)

        tk.Label(main, textvariable=self.stats_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(10, 0))
        tk.Label(main, textvariable=self.status_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(4, 0))

    def _on_source_click(self, event) -> None:
        scale_x = self.source_width / float(self.source_preview_size)
        scale_y = self.source_height / float(self.source_preview_size)
        anchor_x = min(max(event.x * scale_x, 0.0), self.source_width - 1.0)
        anchor_y = min(max(event.y * scale_y, 0.0), self.source_height - 1.0)
        self.rotate_anchor_x_var.set(round(anchor_x))
        self.rotate_anchor_y_var.set(round(anchor_y))
        self._schedule_preview_update()

    def _on_control_change(self, _value=None) -> None:
        self._schedule_preview_update()

    def _schedule_preview_update(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(120, self._update_preview)

    def _draw_source_preview(self) -> None:
        image = Image.fromarray(self.frame_rgb).resize(
            (self.source_preview_size, self.source_preview_size), Image.Resampling.BILINEAR
        )
        self.source_photo = ImageTk.PhotoImage(image)
        self.source_canvas.delete("all")
        self.source_canvas.create_image(0, 0, anchor=tk.NW, image=self.source_photo)

        scale_x = self.source_preview_size / float(self.source_width)
        scale_y = self.source_preview_size / float(self.source_height)
        segment = []
        for point in self.current_outline:
            if point is None:
                if len(segment) >= 2:
                    if len(segment) >= 4:
                        segment.extend(segment[:2])
                    self.source_canvas.create_line(*segment, fill="#00ff66", width=2)
                segment = []
                continue

            px = point[0] * scale_x
            py = point[1] * scale_y
            segment.extend([px, py])

        if len(segment) >= 2:
            if len(segment) >= 4:
                segment.extend(segment[:2])
            self.source_canvas.create_line(*segment, fill="#00ff66", width=2)

        x = float(self.rotate_anchor_x_var.get()) * scale_x
        y = float(self.rotate_anchor_y_var.get()) * scale_y
        size = 10
        self.source_canvas.create_line(x - size, y, x + size, y, fill="red", width=2)
        self.source_canvas.create_line(x, y - size, x, y + size, fill="red", width=2)

    def _update_preview(self) -> None:
        self.after_id = None
        frame_index = int(self.frame_index_var.get())
        if frame_index != self.current_frame_index:
            self._load_frame(frame_index)

        rotate_anchor_x = float(self.rotate_anchor_x_var.get())
        rotate_anchor_y = float(self.rotate_anchor_y_var.get())
        shift_x = float(self.shift_x_var.get())
        shift_y = float(self.shift_y_var.get())
        roll_degrees = float(self.roll_degrees_var.get())
        focal_scale = float(self.focal_scale_var.get())

        output_rgb, stats = remap_frame(
            self.frame_bgr,
            self.camera_params,
            rotate_anchor_x,
            rotate_anchor_y,
            shift_x,
            shift_y,
            roll_degrees,
            focal_scale,
            self.output_width,
            self.output_height,
        )
        self.current_outline = extract_view_outline(
            self.camera_params,
            rotate_anchor_x,
            rotate_anchor_y,
            shift_x,
            shift_y,
            roll_degrees,
            focal_scale,
            self.output_width,
            self.output_height,
        )
        output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)
        self.output_photo = ImageTk.PhotoImage(Image.fromarray(output_rgb))
        self.output_label.configure(image=self.output_photo)
        self._draw_source_preview()

        self.stats_var.set(
            "frame={}  rotate=({:.1f}, {:.1f})  shift=({:.1f}, {:.1f})  roll={:.1f}  focal_scale={:.3f}  "
            "valid_ratio={:.6f}  coverage x:[{:.1f}, {:.1f}] y:[{:.1f}, {:.1f}]".format(
                frame_index,
                rotate_anchor_x,
                rotate_anchor_y,
                shift_x,
                shift_y,
                roll_degrees,
                focal_scale,
                stats["valid_ratio"],
                stats["min_x"],
                stats["max_x"],
                stats["min_y"],
                stats["max_y"],
            )
        )

    def _auto_fit(self) -> None:
        rotate_anchor_x = float(self.rotate_anchor_x_var.get())
        rotate_anchor_y = float(self.rotate_anchor_y_var.get())
        shift_x = float(self.shift_x_var.get())
        shift_y = float(self.shift_y_var.get())
        roll_degrees = float(self.roll_degrees_var.get())
        scale = find_full_view_scale(
            self.camera_params,
            rotate_anchor_x,
            rotate_anchor_y,
            shift_x,
            shift_y,
            roll_degrees,
            self.output_width,
            self.output_height,
        )
        self.focal_scale_var.set(scale)
        self.status_var.set("Auto-fit completed. Review the preview, then click Confirm / Save.")
        self._schedule_preview_update()

    def _save_params(self) -> None:
        payload = {
            "rotate_anchor_x": float(self.rotate_anchor_x_var.get()),
            "rotate_anchor_y": float(self.rotate_anchor_y_var.get()),
            "shift_x": float(self.shift_x_var.get()),
            "shift_y": float(self.shift_y_var.get()),
            "roll_degrees": float(self.roll_degrees_var.get()),
            "focal_scale": float(self.focal_scale_var.get()),
            "output_width": self.output_width,
            "output_height": self.output_height,
        }
        save_anchor_params(self.params_path, payload)
        self.status_var.set("Saved parameters to {}".format(self.params_path.name))

    def _on_close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI tool for tuning SEUCM anchor and focal scale.")
    parser.add_argument("input_video", type=Path, help="Path to the input video")
    parser.add_argument("calibration_json", type=Path, help="Path to the calibration JSON file")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=Path("undistort_params_640x480.json"),
        help="Path to the saved parameter JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = AnchorGui(root, args.input_video, args.calibration_json, args.params_json)
    root.mainloop()


if __name__ == "__main__":
    main()
