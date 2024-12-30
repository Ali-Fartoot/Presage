from abc import ABC, abstractmethod
from openai import OpenAI
import base64
import cv2
from PIL import Image
import PIL
from lang_sam import LangSAM
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path


class ImageProcessingError(Exception):
    """Raised when there's an error processing the image."""
    pass

class ModelInferenceError(Exception):
    """Raised when there's an error during model inference."""
    pass

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    temperature: float = 0.9
    max_tokens: int = 500
    n: int = 1
    stop: Optional[str] = None

class ImageProcessor:
    """Utility class for image processing operations."""
    @staticmethod
    def to_base64(file_path: Union[str, Path]) -> str:
        try:
            with open(file_path, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise ImageProcessingError(f"Failed to convert image to base64: {e}")

class LLMAgent(ABC):
    """Abstract base class for LLM agents."""
    def __init__(self, message: Optional[List[Dict[str, Any]]] = None):
        self.message_template = message or []
        try:
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="llama.cpp")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    @abstractmethod
    def infer(self, image: str, config: ModelConfig) -> str:
        """Abstract method for model inference."""
        pass

    def _update_message_template(self, base64_image: str) -> None:
        """Update message template with base64 image."""
        try:
            self.message_template[1]["content"][0]["image_url"]["url"] = base64_image
        except (IndexError, KeyError) as e:
            raise ValueError(f"Invalid message template structure: {e}")

class Fallback(LLMAgent):
    """Agent to check if given image contains a palm."""
    def __init__(self):
        message = [
            {
                "role": "system",
                "content": "You are an assistant who should classify if the given image contains a palm or not."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": None}},
                    {"type": "text", "text": "Is the given image consist of palm or not? return Yes or No in one word."}
                ]
            }
        ]
        super().__init__(message)

    def infer(self, image: str, config: Optional[ModelConfig] = None) -> str:
        try:
            config = config or ModelConfig(temperature=0.1, max_tokens=10)
            base64_image = ImageProcessor.to_base64(image)
            self._update_message_template(base64_image)

            response = self.client.chat.completions.create(
                model="local-model",
                messages=self.message_template,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n=config.n,
                stop=config.stop
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise ModelInferenceError(f"Inference failed: {e}")

class PresageAgent(LLMAgent):
    """Agent for future predictions based on hand images."""
    def __init__(self):
        message = [
            {
                "role": "system",
                "content": "You are an assistant who should creatively presage the future of a person based on their hand image."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": None}},
                    {"type": "text", "text": "Use your creativity to presage their future."}
                ]
            }
        ]
        super().__init__(message)

    def infer(self, image: str, config: Optional[ModelConfig] = None) -> str:
        try:
            config = config or ModelConfig()
            base64_image = ImageProcessor.to_base64(image)
            self._update_message_template(base64_image)

            response = self.client.chat.completions.create(
                model="local-model",
                messages=self.message_template,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n=config.n,
                stop=config.stop
            )
            return response.choices[0].message.content

        except Exception as e:
            raise ModelInferenceError(f"Inference failed: {e}")

class SegmentorAgent:
    """Agent for image segmentation."""
    def __init__(self):
        try:
            self.model = LangSAM()
        except Exception as e:
            raise ModelInferenceError(f"Failed to initialize LangSAM: {e}")

    def __call__(self, image: PIL.Image, text_prompt: Optional[str] = None) -> Any:
        try:
            text_prompt = text_prompt or "Detect and segment the hand's surface"
            return self.model.predict([image], [text_prompt])
        except Exception as e:
            raise ModelInferenceError(f"Segmentation failed: {e}")

class HandLinesDetector:
    """Detector for hand lines in images."""
    @staticmethod
    def process_image(image: np.ndarray) -> np.ndarray:
        """Process image for line detection."""
        try:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.GaussianBlur(img, (9, 9), 0)
            return cv2.Canny(img, 40, 80)
        except Exception as e:
            raise ImageProcessingError(f"Image processing failed: {e}")

    def __call__(self, image_path: str) -> np.ndarray:
        try:
            original = cv2.imread(image_path)
            if original is None:
                raise ImageProcessingError("Failed to load image")

            img = self.process_image(original)
            lined = np.copy(original) * 0

            lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, np.array([]), 50, 20)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255))

            output = cv2.addWeighted(original, 0.8, lined, 1, 0)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise ImageProcessingError(f"Hand line detection failed: {e}")

# Usage Example
def main():
    try:
        # Initialize agents
        fallback_agent = Fallback()
        presage_agent = PresageAgent()
        segmentor_agent = SegmentorAgent()
        hand_lines_detector = HandLinesDetector()

        # Process image
        image_path = "path/to/image.jpg"
        
        # Check if image contains palm
        is_palm = fallback_agent.infer(image_path)
        if is_palm.lower() == "yes":
            # Proceed with analysis
            future_prediction = presage_agent.infer(image_path)
            hand_lines = hand_lines_detector(image_path)
            

    except Exception as e:
        raise

