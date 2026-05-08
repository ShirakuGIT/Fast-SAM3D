# Import required libraries
import torch
from PIL import Image
import requests
from transformers import Sam3Processor, Sam3Model
import numpy as np
import os

# 1. Load Model and Processor
model = Sam3Model.from_pretrained("facebook/sam3", device_map="cuda")
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 2. Load Your Image
image_url = "input_data/rgb.png"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open(image_url).convert("RGB")
original_width, original_height = image.size

print(f"Loaded Image: {original_width}x{original_height}")

# 3. Pre-process Image and Compute Once (Good Performance Practice)
img_inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.no_grad():
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

# 4. Define Your Text Prompts
text_prompts = [
    "tall slim energy drink can with vertical blue red and green stripes and zero sugar text",
    "short peanut tin can with red lid and yellow blue label",
    "black water bottle",
    "coffee jar",
    "tape measure",

] # Add your own prompts
output_dir = "sam3_segmentation_results"
os.makedirs(output_dir, exist_ok=True)

# 5. Run Inference Loop for Prompts
for prompt in text_prompts:
    print(f"\nProcessing prompt: '{prompt}'")
    
    # Process the text prompt
    text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    
    # Run the model with the cached vision embeddings
    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds, **text_inputs)
    
    # Post-process the results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,          # Confidence threshold
        mask_threshold=0.5,     # Threshold to binarize masks
        target_sizes=[(original_height, original_width)]  # Resize back to original
    )[0] # [0] because we have a single image
    
    # Check if any objects were found
    if len(results['masks']) == 0:
        print(f"  No objects found for prompt: '{prompt}'")
        continue
    
    print(f"  Found {len(results['masks'])} object(s).")
    
    # 6. Save Results
    for i, (mask, score, bbox) in enumerate(zip(results['masks'], results['scores'], results['boxes'])):
        # ----- Save mask (tensor -> PIL Image) -----
        mask_tensor = mask.cpu()
        if mask_tensor.dtype == torch.bool:
            mask_np = mask_tensor.numpy().astype(np.uint8) * 255
        else:
            # Apply threshold 0.5 to convert probabilities to binary
            mask_np = (mask_tensor > 0.5).numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np)
        mask_filename = os.path.join(output_dir, f"mask_{prompt}_{i}.png")
        mask_img.save(mask_filename)

        # ----- Save crop from bounding box -----
        bbox_list = bbox.cpu().tolist()          # tensor -> list
        crop = image.crop(bbox_list)
        crop_filename = os.path.join(output_dir, f"crop_{prompt}_{i}.png")
        crop.save(crop_filename)

        print(f"    Object {i} (score: {score:.3f}) -> mask: {mask_filename}, crop: {crop_filename}")