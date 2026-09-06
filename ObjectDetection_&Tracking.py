import argparse
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection and tracking with YOLOv8."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. 0) or path to a video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights (auto-downloaded on first use if not found locally).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm configuration to use.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for detections (0.0 - 1.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to save the annotated output video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window (still saves the output video).",
    )
    return parser.parse_args()


def resolve_source(source):
    """Treat a purely numeric source string as a webcam index."""
    return int(source) if source.isdigit() else source


def main():
    args = parse_args()
    source = resolve_source(args.source)

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        sys.exit(1)

    # Probe the source once to get video properties for the output writer.
    probe_cap = cv2.VideoCapture(source)
    if not probe_cap.isOpened():
        print(f"Error: could not open video source '{args.source}'.")
        sys.exit(1)

    fps = probe_cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 20.0
    width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    probe_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: could not open output video writer for '{args.output}'.")
        sys.exit(1)q

    print("Starting detection and tracking. Press 'q' in the preview window to quit.")

    frame_count = 0
    try:
        results_stream = model.track(
            source=source,
            conf=args.conf,
            tracker=args.tracker,
            stream=True,
            persist=True,
            verbose=False,
        )

        for result in results_stream:
            annotated_frame = result.plot()  # draws boxes, class labels, and track IDs
            writer.write(annotated_frame)
            frame_count += 1

            if not args.no_display:
                cv2.imshow("Object Detection and Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        writer.release()
        cv2.destroyAllWindows()
        print(f"Done. Processed {frame_count} frames. Annotated video saved to '{args.output}'.")


if __name__ == "__main__":
    main()