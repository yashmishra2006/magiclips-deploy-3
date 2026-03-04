import sys
import os
import argparse

# Add current directory to path if needed (if script is inside lambda_function/)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lambda_function.video_processing import detect_faces_and_crop
except ImportError:
    # If running from inside lambda_function folder
    from video_processing import detect_faces_and_crop

def main():
    parser = argparse.ArgumentParser(description="Test video processing locally.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("--output", default="output_test.mp4", help="Path for the output video")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=10.0, help="End time in seconds")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"--- Starting Local Processing ---")
    print(f"Input:  {args.input}")
    print(f"Range:  {args.start}s to {args.end}s")
    print(f"Output: {args.output}")
    print(f"---------------------------------")

    result = detect_faces_and_crop(
        video_path=args.input,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end
    )

    if result and os.path.exists(result):
        print(f"\nSUCCESS! Created: {os.path.abspath(result)}")
    else:
        print("\nFAILED: Processing did not return a valid output path.")

if __name__ == "__main__":
    main()
