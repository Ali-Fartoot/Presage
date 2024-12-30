from abc import ABC, abstractmethod
from openai import OpenAI
import base64
import cv2
from PIL import Image
import PIL
from lang_sam import LangSAM
import numpy as np


class ImageProcessingError(Exception):
    """Raised when there's an error processing the image."""
    pass

class ModelInferenceError(Exception):
    """Raised when there's an error during model inference."""
    pass


class LLMAgent(ABC):
    """
    Abstract base class for LLM agents that handle image processing and inference.
    """
    def __init__(self, message):
        self.message_template = message or []
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="llama.cpp")

    
    def _image_to_base64_data_uris(self, file_path: str) -> str:
        """
        Convert image file to base64 data URI.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded data URI
        """
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{base64_data}"
    
    @abstractmethod
    def infer(self, 
              temperature: float = 0.9,
              max_token: int = 500,
              n: int = 1,
              stop: str = None) -> any:
        """
        Abstract method for model inference.
        
        Args:
            temperature (float): Sampling temperature
            max_token (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            stop (str): Stop sequence
            
        Returns:
            Implementation dependent return type
        """
        pass

# Fallback Agent to check is given image comsist of palm or not
class Fallback(LLMAgent):
    def __init__(self, message: list[dict]):
        super().__init__(message)
        self.message_template = message or [
            {"role": "system", "content": "You are an assistant who should classify if the given image contains a palm or not."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": None }},
                    {"type" : "text", "text": "Is the given image consist of palm or not? return Yes or No in one word."}
                ]
            }
        ] 


    def infer(self, image, temperature = 0.1, max_token = 10, n = 1, stop = None):

        base64_image = self._image_to_base64_data_uris(image)
        self.message_template[1]["content"][0]["image_url"]["url"] = base64_image

        response = self.client.chat.completions.create(
        model="local-model",
        messages=self.message_template,
        temperature=temperature,
        max_tokens=max_token,
        n=n,
        stop=stop    
        )

        return response.choices[0].message.content
    

# Presage Agent
class PresageAgent(LLMAgent):
    def __init__(self, message):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": "You are an assistant who specializes in creative fortune-telling by analyzing images of people's hands."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": None}},
                    {"type": "text", "text": "Please provide a creative interpretation of this person's future based on their hand image."}
                ]
            }
        ]

    def infer(self, image, temperature = 0.9, max_token = 1000, n = 1, stop = None):

        base64_image = self._image_to_base64_data_uris(image)
        self.message_template[1]["content"][0]["image_url"]["url"] = base64_image

        response = self.client.chat.completions.create(
        model="local-model",
        messages=self.message_template,
        temperature=temperature,
        max_tokens=max_token,
        n=n,
        stop=stop    
        )

        return response.choices[0].message.content
    

class SegmentorAgent:
    """Segment hand of given image for acuurate"""
    def __call__(self, image: PIL.Image, text_prompt: str):
        self.model = LangSAM()
        text_prompt = text_prompt or "Detect and segment the hand's surface"
        # image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        mask = self.model.predict([image], [text_prompt])
        mask = mask.squeeze()

        if len(mask.shape) == 3:
            mask = mask[0]

        white_background = np.ones_like(image_array) * 255
        processed_image = np.where(mask[..., None], image_array, white_background)
        processed_image = processed_image.astype('uint8')

        return Image.fromarray(processed_image)


class HandLinesDetector:
    """Detector for hand lines in images."""
    
    def __init__(self, angle_threshold=15, distance_threshold=25):
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold

    @staticmethod
    def merge_similar_lines(lines, angle_threshold=10, distance_threshold=20):
        """Merge similar lines based on angle and distance"""
        if len(lines) == 0:
            return []
        
        merged_lines = []
        groups = []
        
        for line in lines:
            (x1, y1), (x2, y2), angle = line
            matched = False
            
            for group in groups:
                ref_line = merged_lines[group[0]]
                ref_angle = ref_line[2]
                
                if abs(angle - ref_angle) < angle_threshold:
                    dist = min(
                        np.sqrt((x1-ref_line[0][0])**2 + (y1-ref_line[0][1])**2),
                        np.sqrt((x2-ref_line[1][0])**2 + (y2-ref_line[1][1])**2)
                    )
                    
                    if dist < distance_threshold:
                        group.append(len(merged_lines))
                        matched = True
                        break
            
            if not matched:
                groups.append([len(merged_lines)])
            
            merged_lines.append(((x1, y1), (x2, y2), angle))
        
        final_lines = []
        for group in groups:
            if len(group) == 1:
                final_lines.append(merged_lines[group[0]])
            else:
                x1_avg = np.mean([merged_lines[i][0][0] for i in group])
                y1_avg = np.mean([merged_lines[i][0][1] for i in group])
                x2_avg = np.mean([merged_lines[i][1][0] for i in group])
                y2_avg = np.mean([merged_lines[i][1][1] for i in group])
                angle_avg = np.mean([merged_lines[i][2] for i in group])
                
                final_lines.append(((int(x1_avg), int(y1_avg)), 
                                  (int(x2_avg), int(y2_avg)), 
                                  angle_avg))
        
        return final_lines

    def process_image(self, image: np.ndarray) -> tuple:
        """Process image for line detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Multi-scale processing
            all_detected_lines = []
            scales = [(7, 7), (9, 9), (11, 11)]
            
            for scale in scales:
                bilateral = cv2.bilateralFilter(enhanced, 11, 85, 85)
                blurred = cv2.GaussianBlur(bilateral, scale, 0)
                
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 15, 3
                )

                kernel_line = np.ones((5,5), np.uint8)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)
                dilated = cv2.dilate(morph, kernel_line, iterations=1)
                edges = cv2.Canny(dilated, 30, 150, apertureSize=3)
                
                lines = cv2.HoughLinesP(
                    edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=30,
                    minLineLength=50,
                    maxLineGap=15
                )
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if length > 50:
                            all_detected_lines.append(((x1, y1), (x2, y2), angle))
            
            return edges, all_detected_lines
            
        except Exception as e:
            raise Exception(f"Image processing failed: {e}")

    def __call__(self, image_path: str) -> np.ndarray:
        try:
            # Read image
            original = cv2.imread(image_path)
            if original is None:
                raise Exception("Failed to load image")

            # Process image and get lines
            _, all_detected_lines = self.process_image(original)
            
            # Create empty image for lines
            final_lines = np.zeros_like(original)
            
            # Merge similar lines
            merged_lines = self.merge_similar_lines(
                all_detected_lines, 
                self.angle_threshold, 
                self.distance_threshold
            )

            # Draw merged lines
            for line in merged_lines:
                (x1, y1), (x2, y2), _ = line
                cv2.line(final_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Post-processing
            kernel_smooth = np.ones((3,3), np.uint8)
            final_lines = cv2.dilate(final_lines, kernel_smooth, iterations=1)
            final_lines = cv2.erode(final_lines, kernel_smooth, iterations=1)

            # Combine with original image
            alpha = 0.7
            beta = 1.0
            gamma = 0
            output = cv2.addWeighted(original, alpha, final_lines, beta, gamma)

            # Final enhancement
            output_enhanced = cv2.convertScaleAbs(output, alpha=1.2, beta=10)
            
            return cv2.cvtColor(output_enhanced, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise Exception(f"Hand line detection failed: {e}")