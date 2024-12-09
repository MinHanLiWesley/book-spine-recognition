import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
from google.cloud import vision
import base64
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def resize_image_to_height(image, target_height=640):
    """
    Resizes an image to a specified height while maintaining aspect ratio.
    
    Args:
        image (np.array): The input image.
        target_height (int): The desired height of the resized image.
    
    Returns:
        np.array: The resized image.
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))

def calculate_overlap(original_width, resized_width, tile_size=640):
    """
    Calculates the overlap needed for image tiling based on the original and resized widths.
    
    Args:
        original_width (int): The width of the original image.
        resized_width (int): The width of the resized image.
        tile_size (int): The size of each tile.
    
    Returns:
        int: The calculated overlap.
    """
    scale_factor = original_width / resized_width
    return int(tile_size * scale_factor * 0.5)

def split_image_horizontally(image, tile_size=640, overlap=50):
    """
    Splits an image into horizontal tiles with specified overlap.
    
    Args:
        image (np.array): The input image.
        tile_size (int): The size of each tile.
        overlap (int): The overlap between tiles.
    
    Returns:
        list: A list of tuples containing the tile and its x-coordinate.
    """
    tiles = []
    h, w, _ = image.shape
    for x in range(0, w - overlap, tile_size - overlap):
        tile = image[:, x:min(x + tile_size, w)]
        tiles.append((tile, x))
    return tiles

def pad_image_width(image, tile_size=640):
    """
    Pads the width of an image to be a multiple of the tile size.
    
    Args:
        image (np.array): The input image.
        tile_size (int): The size of each tile.
    
    Returns:
        np.array: The padded image.
    """
    h, w, _ = image.shape
    pad_w = (tile_size - w % tile_size) % tile_size
    return cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def calculate_overlap_percentage(boxA, boxB):
    """
    Calculates the percentage of overlap between two bounding boxes.
    
    Args:
        boxA (tuple): The first bounding box (x1, y1, x2, y2).
        boxB (tuple): The second bounding box (x1, y1, x2, y2).
    
    Returns:
        float: The overlap percentage.
    """
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    if xA1 > xA2: xA1, xA2 = xA2, xA1
    if yA1 > yA2: yA1, yA2 = yA2, yA1
    xI1, yI1 = max(xA1, xB1), max(yA1, yB1)
    xI2, yI2 = min(xA2, xB2), min(yA2, yB2)
    interWidth, interHeight = max(0, xI2 - xI1), max(0, yI2 - yI1)
    interArea = interWidth * interHeight
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    return interArea / float(boxAArea)

def predict_image_with_horizontal_split(model, image, conf_threshold=0.30, target_height=640):
    """
    Predicts objects in an image by splitting it horizontally and using a model.
    
    Args:
        model: The object detection model.
        image (np.array): The input image.
        conf_threshold (float): The confidence threshold for predictions.
        target_height (int): The target height for resizing the image.
    
    Returns:
        tuple: Annotated image and list of final bounding boxes.
    """
    original_height, original_width = image.shape[:2]
    resized_image = resize_image_to_height(image, target_height)
    padded_image = pad_image_width(resized_image)
    overlap = calculate_overlap(original_width, padded_image.shape[1])
    tiles = split_image_horizontally(padded_image, overlap=overlap)
    results = []

    for tile, x in tiles:
        result = model.predict(source=tile, conf=conf_threshold, save=False, show=False, retina_masks=True)[0]
        for box in result.boxes:
            confidence = box.conf
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            adjusted_box = (x1 + x, y1, x2 + x, y2)
            results.append((confidence, *adjusted_box))

    final_boxes = merge_boxes(results)
    annotated_frame, original_boxes = annotate_frame(padded_image, final_boxes, original_width, original_height, resized_image.shape[1], resized_image.shape[0])
    
    return annotated_frame, original_boxes

def merge_boxes(results):
    """
    Merges bounding boxes based on confidence levels and overlap.
    
    Args:
        results (list): List of tuples containing confidence and bounding box coordinates.
    
    Returns:
        list: Sorted list of final bounding boxes.
    """
    green_boxes, red_boxes, all_boxes = [], [], []
    for confidence, x1, y1, x2, y2 in results:
        if confidence > 0.50:
            green_boxes.append((x1, y1, x2, y2))
        elif 0.30 <= confidence <= 0.50:
            red_boxes.append((x1, y1, x2, y2))
    
    # # Filter out red boxes that overlap significantly with any green box
    # for (rx1, ry1, rx2, ry2) in red_boxes:
    #     if not any(calculate_overlap_percentage((rx1, ry1, rx2, ry2), (gx1, gy1, gx2, gy2)) > 0.90 for (gx1, gy1, gx2, gy2) in green_boxes):
    #         all_boxes.append((rx1, ry1, rx2, ry2))

    all_boxes.extend(green_boxes)
    all_boxes.extend(red_boxes)
    all_boxes = sorted(all_boxes, key=lambda x: (x[0], x[1]))

    # Merge boxes that overlap significantly
    tmp = all_boxes.copy()
    for box in all_boxes:
        if box in tmp:
            overlap_boxes = [box2 for box2 in tmp if calculate_overlap_percentage(box, box2) > 0.95]
            if len(overlap_boxes) == 1:
                continue
            else:
                min_x1 = min([box2[0] for box2 in overlap_boxes])
                min_y1 = min([box2[1] for box2 in overlap_boxes])
                max_x2 = max([box2[2] for box2 in overlap_boxes])
                max_y2 = max([box2[3] for box2 in overlap_boxes])
                tmp.append((min_x1, min_y1, max_x2, max_y2))
                # remove the overlap boxes from all_boxes
                for box2 in overlap_boxes:
                    if box2 in tmp:
                        tmp.remove(box2) 
    
    final_boxes = sorted(tmp, key=lambda x: (x[0], x[1]))

    return final_boxes

def annotate_frame(padded_image, final_boxes, original_width, original_height, resized_width, resized_height):
    """
    Annotates an image with bounding boxes and resizes it to the original dimensions.
    
    Args:
        padded_image (np.array): The padded image.
        final_boxes (list): List of final bounding boxes.
        original_width (int): The original width of the image.
        original_height (int): The original height of the image.
        resized_width (int): The width of the resized image.
        resized_height (int): The height of the resized image.
    
    Returns:
        np.array: The annotated image.
    """
    # Initialize annotated_frame as a copy of the padded_image
    annotated_frame = padded_image.copy()
    rainbow_colors = [(148, 0, 211), (75, 0, 130), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0)]
    
    # Draw each bounding box on the image
    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        color = rainbow_colors[i % len(rainbow_colors)]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # Rescale boxes to original dimensions
    scale_x = original_width / float(resized_width)
    scale_y = original_height / float(resized_height)
    original_boxes = [(int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)) for (x1, y1, x2, y2) in final_boxes]
    
    # Resize annotated frame to original size
    annotated_frame = cv2.resize(annotated_frame[:, :resized_width], (original_width, original_height))
    
    # Show number of boxes
    print(f"Number of boxes: {len(original_boxes)}")
    
    # Draw annotated frame using matplotlib
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Annotated Frame ({len(original_boxes)} boxes)")
    plt.axis('off')
    plt.show()
    

    return annotated_frame, original_boxes

def detect_text_google_vision(image):
    """
    Detects text in an image using Google Vision API.
    
    Args:
        image (np.array): The input image.
    
    Returns:
        str: The detected text.
    """
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image to bytes")
    base64_image = base64.b64encode(encoded_image).decode('utf-8')
    vision_image = vision.Image(content=base64_image)
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def main():
    """
    Main function to load model, process images, and detect text.
    """
    # Define paths for model weights and test images
    weights_path = "../model/weights/best.pt"
    test_dir = "../testImage/crops"
    output_dir = "../output"  # Directory to save outputs

    # Load the YOLO model with specified weights
    model = YOLO(weights_path)
    model.to("cuda")  # Move model to GPU for faster processing
    print(model.info())  # Print model information

    # Check if the model weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights not found at {weights_path}")

    # List all image files in the test directory
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        raise FileNotFoundError("No test images found")

    # Initialize EasyOCR reader for text detection
    reader = easyocr.Reader(['en'])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the test directory
    for image_name in test_images:
        image_path = os.path.join(test_dir, image_name)
        print(f"\nProcessing image: {image_name}")
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]
        
        # Predict objects in the image using the model
        annotated_frame, bounding_boxes = predict_image_with_horizontal_split(model, image)

        
        # Rescale bounding boxes to original image size
        scale_x = original_width / float(annotated_frame.shape[1])
        scale_y = original_height / float(annotated_frame.shape[0])
        original_boxes = [(int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)) for (x1, y1, x2, y2) in bounding_boxes]

        # Prepare to store text and bounding box data
        text_data = []

        # Create a directory for the current image's outputs
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
        os.makedirs(image_output_dir, exist_ok=True)

        # Process each bounding box detected in the image
        for i, bounding_box in enumerate(original_boxes):
            x1, y1, x2, y2 = bounding_box
            cropped_image = image[y1:y2, x1:x2]  # Crop the image to the bounding box
            
            # Detect text in the cropped image using Google Vision API
            text = detect_text_google_vision(cropped_image)
            # Replace newlines with spaces
            text = text.replace('\n', ' ')
            text_data.append((bounding_box, text))  # Store index, bounding box, and text

            # Save the cropped image with bounding box number inside the image-specific directory
            cropped_image_path = os.path.join(image_output_dir, f"box_{i}.png")
            cv2.imwrite(cropped_image_path, cropped_image)

            print(f"Cropped image {i}: {text}")

        # Annotate the original image with bounding boxes and text
        for i, (x1, y1, x2, y2) in enumerate(original_boxes):
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{i}: {text_data[i][1]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save the annotated image in the image-specific directory
        annotated_image_path = os.path.join(image_output_dir, f"annotated_{image_name}")
        cv2.imwrite(annotated_image_path, image)

        # Save text data to a file in the image-specific directory
        text_file_path = os.path.join(image_output_dir, f"text_data.txt")
        with open(text_file_path, 'w') as f:
            for i, (bbox, text) in enumerate(text_data):
                f.write(f"Box {i}: {bbox}, Text: {text}\n")

if __name__ == "__main__":
    main()