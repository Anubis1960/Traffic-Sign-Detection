import argparse
from signdetect import SignDetector

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        help='Path to image/video file')
    parser.add_argument('--model', type=str, default='model.keras',
                        help='Cnn model path')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output path for video file')

    args = parser.parse_args()

    detector = SignDetector(args.model)

    detector.detect_traffic_signs(
        input=args.input,
        output=args.output
    )


if __name__ == "__main__":
    main()