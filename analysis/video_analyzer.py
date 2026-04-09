import os
import csv
import time
from collections import Counter

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class VideoLaneAnalyzer:
    def __init__(self, model_path='tusimple_best.pth'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.input_w = 800
        self.input_h = 288

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frame_bgr):
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        x = (x - self.mean) / self.std
        return x

    def detect_lanes(self, frame_bgr, thr=0.30):
        orig_h, orig_w = frame_bgr.shape[:2]

        x = self.preprocess(frame_bgr)

        with torch.no_grad():
            pred = torch.sigmoid(self.model(x))[0, 0].detach().cpu().numpy()

        prob_small = pred.copy()
        mask_small = (pred > thr).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)

        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        prob = cv2.resize(prob_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return mask, prob

    def calculate_metrics(self, mask):
        h, w = mask.shape[:2]
        roi = mask[int(h * 0.55):, :]  # нижняя часть кадра важнее для полос

        lane_pixels = np.count_nonzero(roi)
        confidence = lane_pixels / roi.size

        moments = cv2.moments(roi)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            frame_center = w // 2
            offset_px = cx - frame_center
            offset_norm = offset_px / frame_center
            steering_angle = float(offset_norm * 30.0)
        else:
            offset_px = 0
            offset_norm = 0.0
            steering_angle = 0.0

        if confidence < 0.01:
            command = "LANES_NOT_VISIBLE"
        elif abs(offset_norm) < 0.05:
            command = "STRAIGHT"
        elif offset_norm < 0:
            command = "RIGHT"
        else:
            command = "LEFT"

        warning = int(abs(offset_norm) > 0.35)

        return {
            "confidence": float(confidence),
            "offset_px": int(offset_px),
            "offset_norm": float(offset_norm),
            "steering_angle": steering_angle,
            "command": command,
            "warning": warning
        }

    def draw_overlay(self, frame, mask, metrics, frame_idx, total_frames, fps):
        out = frame.copy()

        overlay = out.copy()
        overlay[mask > 127] = (0, 0, 255)
        out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)

        h, w = out.shape[:2]
        center_x = w // 2
        cv2.line(out, (center_x, 0), (center_x, h), (0, 255, 255), 2)

        roi_y = int(h * 0.55)
        cv2.line(out, (0, roi_y), (w, roi_y), (255, 255, 0), 1)

        txt_color = (0, 255, 0) if metrics["warning"] == 0 else (0, 0, 255)

        cv2.putText(out, f'Frame: {frame_idx+1}/{total_frames}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(out, f'FPS(src): {fps:.1f}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f'Cmd: {metrics["command"]}', (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, txt_color, 2)
        cv2.putText(out, f'Offset: {metrics["offset_norm"]:+.3f}', (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f'Angle: {metrics["steering_angle"]:+.1f} deg', (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f'Conf: {metrics["confidence"]:.3f}', (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if metrics["warning"]:
            cv2.putText(out, 'WARNING: lane departure risk', (20, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return out

    def save_csv(self, rows, csv_path):
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        return df

    def save_plot(self, df, plot_path):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(df["frame"], df["offset_norm"], linewidth=1.2)
        axes[0].axhline(0, linestyle="--")
        axes[0].set_title("Lane offset over time")
        axes[0].set_ylabel("Offset norm")

        axes[1].plot(df["frame"], df["confidence"], linewidth=1.2)
        axes[1].set_title("Lane confidence over time")
        axes[1].set_ylabel("Confidence")
        axes[1].set_xlabel("Frame")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def save_summary(self, df, summary_path, video_path, output_video_path):
        cmd_counts = Counter(df["command"])
        summary = []
        summary.append("VIDEO LANE ANALYSIS REPORT")
        summary.append("")
        summary.append(f"Input video: {video_path}")
        summary.append(f"Output video: {output_video_path}")
        summary.append(f"Frames analyzed: {len(df)}")
        summary.append(f"Average confidence: {df['confidence'].mean():.4f}")
        summary.append(f"Max abs offset: {df['offset_norm'].abs().max():.4f}")
        summary.append(f"Average abs offset: {df['offset_norm'].abs().mean():.4f}")
        summary.append(f"Warnings: {int(df['warning'].sum())}")
        summary.append("")
        summary.append("Command distribution:")
        for k, v in cmd_counts.items():
            summary.append(f"  {k}: {v}")
        summary.append("")
        summary.append("Interpretation:")
        summary.append("- High confidence + low offset means stable lane tracking.")
        summary.append("- Frequent warnings suggest unstable lane fit or unsuitable input video.")
        summary.append("- If output is noisy, use a video with clear road markings and dashboard/dashcam perspective.")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))

    def analyze_video(self, video_path, output_video_path='analyzed_video.mp4',
                      csv_path='lane_metrics.csv', plot_path='lane_plot.png',
                      summary_path='lane_summary.txt'):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        rows = []
        frame_idx = 0
        t0 = time.time()

        print(f"Start: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mask, prob = self.detect_lanes(frame)
            metrics = self.calculate_metrics(mask)

            row = {
                "frame": frame_idx,
                "confidence": metrics["confidence"],
                "offset_px": metrics["offset_px"],
                "offset_norm": metrics["offset_norm"],
                "steering_angle": metrics["steering_angle"],
                "command": metrics["command"],
                "warning": metrics["warning"]
            }
            rows.append(row)

            vis = self.draw_overlay(frame, mask, metrics, frame_idx, total_frames, fps)
            out.write(vis)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames}")

        cap.release()
        out.release()

        df = self.save_csv(rows, csv_path)
        self.save_plot(df, plot_path)
        self.save_summary(df, summary_path, video_path, output_video_path)

        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s")
        print(f"Saved video: {output_video_path}")
        print(f"Saved csv: {csv_path}")
        print(f"Saved plot: {plot_path}")
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    analyzer = VideoLaneAnalyzer(model_path='tusimple_best.pth')
    analyzer.analyze_video(
        video_path='test_video.mp4',
        output_video_path='analyzed_video.mp4',
        csv_path='lane_metrics.csv',
        plot_path='lane_plot.png',
        summary_path='lane_summary.txt'
    )