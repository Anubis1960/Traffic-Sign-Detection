import cv2
import numpy as np
import os
import tensorflow as tf
import cv2.typing
import sys
import logging
from utils import classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)


class SignDetector:
    def __init__(self, model_path: str = None):

        self.model = None
        self.confidence_threshold = 0.7
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                logging.info(f"Loaded model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load model from {model_path}: {e}")
                self.model = None

        # Define color ranges for sign detection
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 150, 50]), np.array([140, 255, 255]))
            ],
            'yellow': [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))
            ]
        }

        # Morphological kernel
        self.kernel = np.ones((5, 5), np.uint8)

    def detect_traffic_signs(self, input: str, output: str) -> None:

        if input.endswith('.mp4'):
            self.process_video(input, output)
        else:
            self.process_image(input)

    def process_video(self, video_path: str, output_path: str) -> None:

        cap = cv2.VideoCapture(video_path)

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            processed_frame = self.process_frame(frame)
            frames.append(processed_frame)

        cap.release()
        cv2.destroyAllWindows()

        self.export_video_from_frames(frames, output_path, 30)

    def export_video_from_frames(self, frames: list, output_path: str, fps: int = 15) -> None:

        if not frames:
            raise ValueError("No frames found in folder.")

        first_frame = frames[0]
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files or 'XVID' for .avi files)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for img in frames:
            video_writer.write(img)

        video_writer.release()
        logging.info(f"Video exported to: {output_path}")

    def process_image(self, image_path: str) -> None:

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Error: Could not load image from {image_path}")
            return

        processed_image = self.process_frame(image)

        cv2.imshow('Traffic Sign', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_frame(self, frame: cv2.typing.MatLike) -> np.ndarray:

        result = frame.copy()
        h, w = frame.shape[:2]

        # Resize for consistent processing if needed
        if h > 800 or w > 800:
            scale = min(800 / h, 800 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            result = cv2.resize(result, (new_w, new_h))

        # Detect potential sign regions
        sign_regions = self.detect_sign_regions(frame)

        # Process each detected region
        for i, (contour, color_type) in enumerate(sign_regions):
            # Crop and preprocess the sign
            cropped_sign, bbox = self.crop_and_preprocess_sign(frame, contour)

            if cropped_sign is not None:
                # Predict using CNN model
                prediction = self.predict_sign(cropped_sign)

                # Draw annotation with prediction
                if prediction is not None:
                    self.draw_annotation(result, contour, color_type, prediction, bbox)

        return result

    def detect_sign_regions(self, image: cv2.typing.MatLike) -> list[tuple[np.ndarray, np.ndarray]]:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        all_contours = []

        for color_name, ranges in self.color_ranges.items():
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < 30000:
                    all_contours.append((cnt, color_name.upper()))

        return all_contours

    def crop_and_preprocess_sign(self, image: cv2.typing.MatLike, contour: cv2.typing.MatLike, border_size: int = 10) -> tuple:

        # Get bounding rectangle with some padding
        x, y, w, h = cv2.boundingRect(contour)

        # Add border
        x1 = max(0, x - border_size)
        y1 = max(0, y - border_size)
        x2 = min(image.shape[1], x + w + border_size)
        y2 = min(image.shape[0], y + h + border_size)

        # Crop the sign
        cropped = image[y1:y2, x1:x2].copy()

        if cropped.size == 0:
            return None, None

        # Preprocess for CNN (30x30 grayscale)
        processed = self.preprocess_for_cnn(cropped)

        return processed, (x1, y1, x2 - x1, y2 - y1)

    def preprocess_for_cnn(self, image: cv2.typing.MatLike) -> np.ndarray:

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to 30x30
        resized = cv2.resize(gray, (30, 30))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add channel dimension (30, 30, 1)
        processed = np.expand_dims(normalized, axis=-1)

        return processed

    def predict_sign(self, preprocessed_image: cv2.typing.MatLike) -> str | None:

        if self.model is None:
            return None

        try:
            # Add batch dimension (1, 30, 30, 1)
            batch_input = np.expand_dims(preprocessed_image, axis=0)

            predictions = self.model.predict(batch_input, verbose=0)

            # Check confidence threshold
            max_confidence = np.max(predictions[0])
            if max_confidence < self.confidence_threshold:
                logging.debug(f"Low confidence prediction: {max_confidence:.3f}")
                return None

            predicted_class = np.argmax(predictions[0])
            prediction = classes[predicted_class]

            if prediction == "UNKNOWN":
                logging.debug(f"Unknown sign")
                return None

            logging.info(f"Prediction: {prediction}")

            return prediction

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def draw_annotation(self, image: cv2.typing.MatLike, contour: cv2.typing.MatLike, color_type: cv2.typing.MatLike, prediction: str, bbox=None) -> None:

        if bbox is None:
            x, y, w, h = cv2.boundingRect(contour)
        else:
            x, y, w, h = bbox

        # Choose color based on sign type
        color_map = {
            'RED': (0, 0, 255),  # BGR: Red
            'BLUE': (255, 0, 0),  # BGR: Blue
            'YELLOW': (0, 255, 255)  # BGR: Yellow
        }

        color = color_map.get(color_type, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw contour
        cv2.drawContours(image, [contour], -1, color, 1)

        # Prepare label text
        label = f"{prediction}"

        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background for text
        text_bg_y1 = max(0, y - text_height - 10)
        text_bg_y2 = y - 5 if y - 5 > 0 else y + text_height + 10

        cv2.rectangle(image,
                      (x, text_bg_y1),
                      (x + text_width + 10, text_bg_y2),
                      color, -1)

        # Draw text
        cv2.putText(image, label, (x + 5, y - 5 if y > 20 else y + 15),
                    font, font_scale, (255, 255, 255), thickness)
