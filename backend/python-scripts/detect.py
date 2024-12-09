import numpy as np
from ultralytics import YOLO
from google.cloud import vision
import os
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
from pathlib import Path
import sys

class BookDetector:
    def __init__(self, model_path: str, gemini_api_key: str, image_path: str = None):
        """
        Initialize the BookDetector with YOLO model and other components.
        """
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize Gemini with system prompt
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define the system prompt
        self.system_prompt = """You are a visual analysis expert specializing in book spine analysis.

        Core Analysis Rules (In Priority Order):
        1. Text Pattern Analysis:
            - [Small Text] + [Large Text] Pattern:
                * If small text is surname-like → it's likely Author
                * Example: Small "KING" + Large "MISERY" → Author: King, Title: Misery
            - Multiple text sizes indicate different elements
            - Consistent text size/style usually belongs together
        
        2. Author Identification:
            - Usually smaller text than title
            - Often at top or bottom of spine
            - Can be surname only or full name
        
        3. Title Identification:
            - Usually largest/most prominent text
            - Often includes articles
            - Forms complete phrase
            - May span multiple lines in same style
        
        4. Special Cases:
            - Well-known series (Harry Potter, etc):
                * If title is clear → can infer famous author
                * Must note inference in reasoning
        
        5. Always Ignore:
            - Marketing text (BESTSELLER, COMING SOON)
            - Publisher names
            - Price/ISBN
            - Series labels (unless part of title)

        Uncertainty Handling:
        - Include "Need Validation" in the front and state the reason in uncertainty notes if:
            * Cannot correctly determine one of book metadata (either title or author are unclear)
            * Text is completely illegible or missing
            * No recognizable patterns match any standard book spine layout
        
        - Always write "None" in uncertainty notes when there are no uncertainties
          (never leave it empty or omit it)
        
        Important: Do NOT mark for validation just because:
            * Image is low quality but text pattern is clear
            * Text is partially blurry but layout pattern is obvious
            * Author/title can be confidently inferred from visible parts
            * Text is at an angle but readable
            * Standard spine layout is recognizable despite image quality
            * OCR text contains minor errors but pattern is clear

        Remember: If you can confidently identify either the title OR author from the visible evidence and spine layout, the entry should be marked as valid. Only mark for validation when NO reliable book information can be extracted."""

        # Start the chat
        self.chat = self.gemini_model.start_chat(history=[
            {
                "role": "user",
                "parts": [self.system_prompt]
            },
            {
                "role": "model",
                "parts": ["I understand my role as a book spine analysis expert. I will follow the decision rules to extract metadata and provide visual descriptions. I'm ready to analyze book spines."]
            }
        ])

        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust workers as needed

        # Add output path setup
        if image_path:
            image_stem = Path(image_path).stem
            self.output_dir = Path("output") / image_stem
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Create crops directory
            self.crops_dir = self.output_dir / "crops"
            self.crops_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
            self.crops_dir = None

        # Create cache directories for OCR and Gemini results
        if self.output_dir:
            self.ocr_cache_dir = self.output_dir / "ocr_cache"
            self.gemini_cache_dir = self.output_dir / "gemini_cache"
            self.ocr_cache_dir.mkdir(exist_ok=True)
            self.gemini_cache_dir.mkdir(exist_ok=True)
        else:
            self.ocr_cache_dir = Path("ocr_cache")
            self.gemini_cache_dir = Path("gemini_cache")
            self.ocr_cache_dir.mkdir(exist_ok=True)
            self.gemini_cache_dir.mkdir(exist_ok=True)



    def detect_and_crop_books(self, image: np.ndarray) -> List[Tuple[np.ndarray, tuple]]:
        """
        Detect and crop books from the image using four-point coordinates, rotate to vertical,
        and sort from left to right.
        """
        
        results = self.model.predict(source=image, conf=0.50, save=False, show=False)[0]
        cropped_books = []
        
        if not results or not hasattr(results, 'obb') or len(results.obb) == 0:
            return []
        
        for box in results.obb:
            try:
                # Get eight points (x,y coordinates) from the box and reshape to 4x2
                points = box.xyxyxyxy.cpu().numpy().reshape(-1, 2)
               
                # Calculate angle and get rotation matrix
                edge1 = points[1] - points[0]
                angle = np.arctan2(edge1[1], edge1[0]) * 180 / np.pi
                
                # Get the center point
                center = np.mean(points, axis=0)
                
                # Get width and height of the rotated rectangle
                width = np.linalg.norm(points[1] - points[0])
                height = np.linalg.norm(points[2] - points[1])
                
                # Skip if width or height is too small
                if width < 10 or height < 10:
                    continue
                
                # Get rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
                
                # Calculate new image size after rotation
                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_width = int(height * sin + width * cos)
                new_height = int(height * cos + width * sin)
                
                # Adjust rotation matrix
                rotation_matrix[0, 2] += new_width/2 - center[0]
                rotation_matrix[1, 2] += new_height/2 - center[1]
                
                # Rotate the whole image
                rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
                
                # Calculate the bounding box in the rotated image
                rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
                x_min, y_min = np.min(rotated_points, axis=0)
                x_max, y_max = np.max(rotated_points, axis=0)
                
                # Ensure valid crop coordinates
                x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
                x_max, y_max = min(rotated_image.shape[1], int(x_max)), min(rotated_image.shape[0], int(y_max))
                
                # Skip if crop dimensions are invalid
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Crop the rotated image
                cropped_image = rotated_image[y_min:y_max, x_min:x_max]
                
                # Store with original bounding box coordinates for reference
                original_bbox = tuple(points.flatten())
                # Don't process the image here, just store the raw crop
                cropped_books.append((cropped_image, original_bbox))
            except Exception:
                continue
        
        if not cropped_books:
            return []
        
        # Sort books by leftmost x coordinate
        cropped_books.sort(key=lambda x: min(x[1][::2]))
        return cropped_books

    def _process_cropped_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process cropped image: resize to 768px max dimension
        No super-resolution needed as it's already done on the full image
        
        Args:
            image (np.ndarray): Input cropped image
            
        Returns:
            np.ndarray: Processed image
        """
        # Resize to 768px max dimension
        height, width = image.shape[:2]
        target_size = 768
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize to target size
        final_image = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        return final_image

    def _get_cache_key(self, image: np.ndarray) -> str:
        """Generate a unique cache key for an image."""
        # Use image hash as cache key
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    async def extract_text_async(self, image: np.ndarray) -> str:
        """Extract text from image using Google Vision API with caching."""
        cache_key = self._get_cache_key(image)
        cache_file = os.path.join(self.ocr_cache_dir, f"{cache_key}.txt")

        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()

        try:
            # Run Vision API call in thread pool
            text = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._vision_api_call,
                image
            )

            # Cache the result
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)

            return text

        except Exception as e:
            return ""

    def _vision_api_call(self, image: np.ndarray) -> str:
        """Make the actual Vision API call."""
        # Convert the image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        
        image_content = encoded_image.tobytes()
        
        # Create vision image object
        vision_image = vision.Image(content=image_content)
        
        # Perform text detection
        response = self.vision_client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        if texts:
            return texts[0].description
        return ""

    def _preprocess_for_gemini(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess image for Gemini input (add padding to make 768x768 square)
        
        Args:
            image (np.ndarray): Input image (already resized to 768px max dimension)
            
        Returns:
            Image.Image: Processed image ready for Gemini
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = rgb_image.shape[:2]
        target_size = 768
        
        # Create black canvas of 768x768
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        x_offset = (target_size - width) // 2
        y_offset = (target_size - height) // 2
        
        # Place image on canvas
        canvas[y_offset:y_offset+height, 
               x_offset:x_offset+width] = rgb_image
        
        # Convert to PIL Image
        return Image.fromarray(canvas)

    async def refine_text(self, text: str, image: np.ndarray) -> Dict[str, str]:
        """
        Refine extracted text using Gemini to identify book metadata with caching.
        """
        # Generate cache key based on both image and OCR text
        cache_key = hashlib.md5(f"{text}{self._get_cache_key(image)}".encode()).hexdigest()
        cache_file = self.gemini_cache_dir / f"{cache_key}.json"

        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                # If cache read fails, continue with normal processing
                pass

        processed_image = self._preprocess_for_gemini(image)
        
        prompt = f"""{self.system_prompt}

        Analyze this book spine with provided OCR text and cropped image:
        OCR Text: {text}
        
        Use standard format to output book metadata:
        TITLE:
        AUTHOR:
        SPINE_APPEARANCE:
        REASONING:
        UNCERTAINTY_NOTES:"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.gemini_model.generate_content([prompt, processed_image])
            )
            
            book_info = await self._parse_gemini_response(response)
            
            if book_info['is_valid']:
                book_info['raw_text'] = text
            
            # Format the response
            result = {
                'author': book_info.get('author', ''),
                'title': book_info.get('title', ''),
                'isValid': book_info.get('is_valid', False),
                'rawText': book_info.get('raw_text', '')
            }

            # Cache the result
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f)
            except Exception:
                # If cache write fails, continue without caching
                pass

            return result

        except Exception as e:
            return {
                'author': '',
                'title': '',
                'isValid': False,
                'rawText': ''
            }

    def _validate_extraction(self, result: Dict[str, str]) -> bool:
        """
        Validate the extraction based on reasoning and uncertainty notes
        
        Args:
            result (Dict[str, str]): Extraction results
            
        Returns:
            bool: True if extraction is valid, False otherwise
        """
        # Check if "Need Validation" flag is present
        if "Need Validation" in result['uncertainty_notes']:
            return False
        
        # Additional validation checks
        high_risk_terms = [
            "cannot determine",
            "unclear",
            "not visible",
            "multiple possibilities",
            "completely obscured",
            "low confidence",
            "uncertain",
            "ambiguous",
            "possibly",
            "might be"
        ]
        
        # Check uncertainty notes for high risk terms
        if any(term in result['uncertainty_notes'].lower() for term in high_risk_terms):
            return False
        
        # Validate reasoning quality
        if len(result['reasoning']) < 20:  # Require more substantial reasoning
            return False
        
        # Validate required fields
        if not result['title'].strip():
            return False
        
        # Special validation for cases with only author
        if not result['author'].strip() and not "series" in result['reasoning'].lower():
            return False
        
        # Check for placeholder or generic responses
        placeholder_terms = ["unknown", "n/a", "not available", "unclear"]
        if any(term in result['title'].lower() for term in placeholder_terms):
            return False
        if any(term in result['author'].lower() for term in placeholder_terms):
            return False
        
        return True

    async def _parse_gemini_response(self, response_content) -> Dict[str, str]:
        """
        Parse a single Gemini response asynchronously
        """
        current_book = {
            'title': '',
            'author': '',
            'spine_appearance': '',
            'reasoning': '',
            'uncertainty_notes': '',
            'is_valid': True  # Start as True, will be validated later
        }
        
        try:
            # Handle different response formats
            if isinstance(response_content, str):
                lines = response_content.split('\n')
            else:
                # If response_content is not a string (e.g., it's a response object)
                lines = response_content.text.split('\n') if hasattr(response_content, 'text') else str(response_content).split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('TITLE:'):
                    current_book['title'] = line.replace('TITLE:', '').strip()
                elif line.startswith('AUTHOR:'):
                    current_book['author'] = line.replace('AUTHOR:', '').strip()
                elif line.startswith('SPINE_APPEARANCE:'):
                    current_book['spine_appearance'] = line.replace('SPINE_APPEARANCE:', '').strip()
                elif line.startswith('REASONING:'):
                    current_book['reasoning'] = line.replace('REASONING:', '').strip()
                elif line.startswith('UNCERTAINTY_NOTES:'):
                    current_book['uncertainty_notes'] = line.replace('UNCERTAINTY_NOTES:', '').strip()

        except Exception as e:
            current_book['uncertainty_notes'] = f"Parsing error: {str(e)}"
            current_book['is_valid'] = False
            return current_book
        
        # Validate the extraction
        current_book['is_valid'] = self._validate_extraction(current_book)
        return current_book

    def draw_annotations(self, image: np.ndarray, boxes: List[np.ndarray], output_path: Path) -> None:
        """
        Draw rainbow-colored bounding boxes with order numbers on the original image.
        
        Args:
            image (np.ndarray): Original image
            boxes (List[np.ndarray]): List of bounding boxes (4 points each, already scaled)
            output_path (Path): Path to save the annotated image
        """
        annotated_image = image.copy()
        
        # Sort boxes from left to right using the minimum x coordinate of each box
        boxes_with_index = [(box, i+1) for i, box in enumerate(boxes)]
        boxes_with_index.sort(key=lambda x: float(np.min(x[0][::2])))  # Convert to float for comparison
        
        # Define rainbow colors (in BGR format)
        rainbow_colors = [
            (0, 0, 255),    # Red
            (0, 127, 255),  # Orange
            (0, 255, 255),  # Yellow
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 0, 127),  # Indigo
            (255, 0, 255)   # Violet
        ]
        
        for idx, (box, order) in enumerate(boxes_with_index):
            # Convert points to integer
            points = box.astype(np.int32)
            
            # Get color from rainbow (cycle through colors if more boxes than colors)
            color = rainbow_colors[idx % len(rainbow_colors)]
            
            # Draw the polygon with thicker line (increased to 8)
            cv2.polylines(annotated_image, [points], True, color, 8)
            
            # Calculate position for order number (above the top-left corner)
            text_x = int(np.min(points[:, 0]))
            text_y = int(np.min(points[:, 1])) - 20
            
            # Draw order number with thicker font (using same color as box)
            cv2.putText(annotated_image, str(order), (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 6)  # Increased thickness to 6
        
        # Save the annotated image
        cv2.imwrite(str(output_path), annotated_image)

    async def process_image_async(self, image_path: str) -> List[Dict]:
        """
        Process a single image and return book information asynchronously.
        """
        # Convert image_path to Path object
        image_path = Path(image_path)
        
        # Check if output directory exists and has cached results
        if self.output_dir and self.output_dir.exists():
            # Generate cache key based on image path and modification time
            cache_key = hashlib.md5(f"{image_path.name}{image_path.stat().st_mtime}".encode()).hexdigest()
            process_cache_file = self.output_dir / "process_cache" / f"{cache_key}.json"
            
            # If cache exists, return it immediately
            if process_cache_file.exists():
                try:
                    with open(process_cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    # If cache read fails, continue with normal processing
                    pass

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Update enhanced path to use output directory
        enhanced_path = self.output_dir / f"{Path(image_path).stem}_enhanced{Path(image_path).suffix}"
        
        original_image = image
        # Try to enhance the image
        if enhance_image_resolution(image_path, enhanced_path):
            enhanced_image = cv2.imread(str(enhanced_path))
            if enhanced_image is not None:  # Check if enhanced image was loaded successfully
                image = enhanced_image

        # Get YOLO results and cropped books
        cropped_books = self.detect_and_crop_books(image)
        
        # Save annotated image if we have any detections
        if cropped_books:
            annotated_path = self.output_dir / f"{Path(image_path).stem}_annotated{Path(image_path).suffix}"
            
            
            # Scale down boxes before passing to draw_annotations
            scale = 2  # Default SR scale
            boxes = [np.array(bbox).reshape(-1, 2) / scale for _, bbox in cropped_books]
            self.draw_annotations(original_image, boxes, annotated_path)
        
        books = []
        ocr_tasks = []
        gemini_tasks = []
        
        # Save cropped images after sorting
        for idx, (book_image, bbox) in enumerate(cropped_books, 1):
            if self.crops_dir:
                crop_path = self.crops_dir / f"{Path(image_path).stem}_crop_{idx}{Path(image_path).suffix}"
                cv2.imwrite(str(crop_path), book_image)
            
            processed_image = self._process_cropped_image(book_image)
            task = asyncio.create_task(self.extract_text_async(processed_image))
            ocr_tasks.append((task, processed_image))
        
        for task, book_image in ocr_tasks:
            try:
                raw_text = await task
                gemini_task = asyncio.create_task(self.refine_text(raw_text, book_image))
                gemini_tasks.append(gemini_task)
            except Exception:
                continue
        
        results = await asyncio.gather(*gemini_tasks, return_exceptions=True)
        final_results = [r for r in results if isinstance(r, dict)]

        # Cache the final results
        try:
            process_cache_file.parent.mkdir(exist_ok=True)
            with open(process_cache_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f)
        except Exception:
            # If cache write fails, continue without caching
            pass

        return final_results

def enhance_image_resolution(input_path, output_path, scale=2):
    """
    Enhance image resolution using RealESRGAN executable
    
    Args:
        input_path (str|Path): Path to input image
        output_path (str|Path): Path to save enhanced image
        scale (int): Upscaling factor (2 or 4)
    Returns:
        bool: True if successful, False otherwise
    """
    realesrgan_path = "../models/realesrgan_portable/realesrgan-ncnn-vulkan.exe"  # Adjust path as needed
    
    command = [
        realesrgan_path,
        '-i', str(input_path),
        '-o', str(output_path),
        '-s', str(scale)
    ]
    
    try:
        result = subprocess.run(command, 
                              check=True, 
                              capture_output=True,
                              text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error enhancing image: {e.stderr}")
        return False



def main():
    import argparse
    import sys
    import json
    from dotenv import load_dotenv

    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Detect and extract information from book images')
    parser.add_argument('image', type=str, help='Path to input image')

    args = parser.parse_args()
    model_path = "../models/yolo_weights/best.pt"
    

    # Set your own api key
    try:
        detector = BookDetector(
            model_path, 
            os.getenv("GEMINI_API"),
            args.image
        )
        books = asyncio.run(detector.process_image_async(args.image))
        
        # Use a unique identifier that won't appear in other logs
        print("BOOK_DETECTION_RESULT:" + json.dumps(books))
        return 0
        
    except Exception as e:
        print("BOOK_DETECTION_RESULT:" + json.dumps({"error": str(e), "success": False}))
        return 1

if __name__ == "__main__":
    sys.exit(main())