from torchvision.transforms import ToTensor
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# Define the detect_wheat_heads function (copying from the notebook)
def detect_wheat_heads(image, faster_rcnn_model, yolo_model, device, confidence_threshold=0.5):
    """
    Performs inference on an image using Faster R-CNN and YOLOv8 models,
    counts detected wheat heads, and returns the count and filtered predictions.

    Args:
        image (PIL.Image or numpy.ndarray): The input image.
        faster_rcnn_model (torchvision.models.detection.FasterRCNN): The trained Faster R-CNN model.
        yolo_model (ultralytics.YOLO): The trained YOLOv8 model.
        device (torch.device): The device to run inference on (cuda or cpu).
        confidence_threshold (float): The confidence threshold for filtering predictions.

    Returns:
        tuple: A tuple containing:
            - int: The total count of detected wheat heads.
            - list: A list of dictionaries, each containing 'boxes' and 'scores'
                    for the filtered predictions from either model.
    """
    # Preprocess the input image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = ToTensor()
    image_tensor = transform(image).to(device)

    all_filtered_predictions = []

    # Pass through the Faster R-CNN model
    if faster_rcnn_model is not None:
        with torch.no_grad():
            frcnn_predictions = faster_rcnn_model([image_tensor])[0]

        frcnn_scores = frcnn_predictions['scores']
        frcnn_boxes = frcnn_predictions['boxes']

        keep_frcnn = frcnn_scores > confidence_threshold
        filtered_frcnn_boxes = frcnn_boxes[keep_frcnn]
        filtered_frcnn_scores = frcnn_scores[keep_frcnn]

        if len(filtered_frcnn_boxes) > 0:
             all_filtered_predictions.append({'model': 'FasterRCNN', 'boxes': filtered_frcnn_boxes, 'scores': filtered_frcnn_scores})

    # Pass through the YOLOv8 model
    if yolo_model is not None:
        yolo_results = yolo_model.predict(source=image, device=device, verbose=False, conf=confidence_threshold)

        if yolo_results and len(yolo_results) > 0:
            yolo_result = yolo_results[0]
            if yolo_result.boxes is not None:
                yolo_boxes = yolo_result.boxes.xyxy
                yolo_scores = yolo_result.boxes.conf
                yolo_classes = yolo_result.boxes.cls

                # Filter by class ID (assuming wheat head is class 0 from our data prep)
                wheat_head_mask = yolo_classes == 0
                filtered_yolo_boxes = yolo_boxes[wheat_head_mask]
                filtered_yolo_scores = yolo_scores[wheat_head_mask]

                if len(filtered_yolo_boxes) > 0:
                    all_filtered_predictions.append({'model': 'YOLOv8', 'boxes': filtered_yolo_boxes, 'scores': filtered_yolo_scores})

    # Count detected wheat heads
    total_wheat_heads = 0
    for prediction_set in all_filtered_predictions:
        total_wheat_heads += len(prediction_set['boxes'])

    return total_wheat_heads, all_filtered_predictions

# Function to draw bounding boxes and text on the image
def draw_detections(image, predictions, num_heads, estimated_grains, estimated_yield_kg):
    image_np = np.array(image)
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: arial.ttf not found, using default font.")

    # Draw bounding boxes
    for prediction_set in predictions:
        model_name = prediction_set['model']
        boxes = prediction_set['boxes'].cpu().numpy()

        if model_name == 'FasterRCNN':
            color = (255, 0, 0) # Red
        elif model_name == 'YOLOv8':
            color = (0, 255, 0) # Green
        else:
            color = (255, 255, 255) # White

        for box in boxes:
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            # Use PIL to draw rectangles for better text integration
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=2)

    # Add text overlays
    text_color = (255, 255, 255) # White text
    text_position_heads = (10, 10)
    text_position_grains = (10, 40)
    text_position_yield = (10, 70)

    draw.text(text_position_heads, f"Detected Heads: {num_heads}", fill=text_color, font=font)
    draw.text(text_position_grains, f"Estimated Grains: {estimated_grains}", fill=text_color, font=font)
    draw.text(text_position_yield, f"Estimated Yield: {estimated_yield_kg:.4f} kg", fill=text_color, font=font)

    return image_pil # Return PIL Image

# --- Streamlit App ---
st.title("Wheat Head Detection and Yield Estimation")

# Load models and set device
@st.cache_resource # Cache the models to avoid reloading on each interaction
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Faster R-CNN
    # Instantiate the model architecture
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2) # Assuming 2 classes: background and wheat head

    # Load your saved state_dict
    try:
        # Adjust the path to your saved .pth file
        faster_rcnn_model.load_state_dict(torch.load('faster_rcnn_model.pth', map_location=device))
        print("Loaded trained Faster R-CNN model.")
    except FileNotFoundError:
        st.warning("Faster R-CNN model file not found. Using pre-trained COCO weights (may not be accurate for wheat heads).")
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    except Exception as e:
        st.error(f"Error loading Faster R-CNN model: {e}. Using pre-trained COCO weights.")
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


    # Load YOLOv8
    try:
        # Adjust the path to your saved .pt file
        yolo_model = YOLO('yolov8_model.pt')
        print("Loaded trained YOLOv8 model.")
    except FileNotFoundError:
        st.warning("YOLOv8 model file not found. Using pre-trained yolov8n.pt (may not be accurate for wheat heads).")
        yolo_model = YOLO('yolov8n.pt')
    except Exception as e:
         st.error(f"Error loading YOLOv8 model: {e}. Using pre-trained yolov8n.pt.")
         yolo_model = YOLO('yolov8n.pt')


    faster_rcnn_model.to(device).eval()
    yolo_model.to(device) # YOLO model's predict handles eval mode internally

    return faster_rcnn_model, yolo_model, device

faster_rcnn_model, yolo_model, device = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    # Adjust confidence threshold if needed
    num_heads, predictions = detect_wheat_heads(image, faster_rcnn_model, yolo_model, device, confidence_threshold=0.5)

    # Display results
    st.subheader("Detection Results")
    st.write(f"Detected Heads: {num_heads}")

    # Calculate and display estimated grains and yield
    estimated_grains = num_heads * 80
    st.write(f"Estimated Grains: {estimated_grains}")

    # Assuming an average grain weight for yield estimation
    average_grain_weight_kg = 0.00005
    estimated_yield_kg = estimated_grains * average_grain_weight_kg
    st.write(f"Estimated Yield: {estimated_yield_kg:.4f} kg")
    st.caption("Note: Estimated yield in kilograms is based on an assumed average grain weight.")


    # Visualize and display the image with bounding boxes
    if num_heads > 0:
        st.subheader("Visualized Detections")
        visualized_image = draw_detections(image, predictions, num_heads, estimated_grains, estimated_yield_kg)
        st.image(visualized_image, caption="Detected Wheat Heads", use_column_width=True)
    else:
        st.info("No wheat heads detected with the current confidence threshold.")