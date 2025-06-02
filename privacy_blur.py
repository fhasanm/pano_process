#!/usr/bin/env python3
"""
privacy_blur.py – 360° anonymiser for equirectangular images.
- Reads *.jpg images from a source directory (and its subdirectories).
- Converts to cube faces, detects faces & plates (YOLOv8).
- Refines detections with SAM.
- Gaussian-blurs masks, re-projects to equirectangular.
- Writes redacted JPGs + a JSON log to a destination directory.
"""
import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
# Ensure SAM is installed and importable. May need to adjust import based on installation.
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Warning: SAM2 not found. Blurring will not use SAM refinement.")
    SAM2ImagePredictor = None # type: ignore

from equilib import equi2cube, cube2equi # Ensure equilib is installed
from tqdm import tqdm
# Assuming utils.py is in the same 'scripts' directory as this file
# and you have a scripts/__init__.py
from utils import ensure_dir_exists # Adjusted to relative import assuming utils is in the same package

def _gaussian_blur_masked_area(img: np.ndarray, mask: np.ndarray, kernel_size: int = 31):
    """Applies Gaussian blur to the image where mask is > 0."""
    if kernel_size % 2 == 0: # Kernel must be odd
        kernel_size +=1
    if mask.any(): # Only apply blur if there's actually a mask
        # Ensure the image is writable if it's a view from another array
        if not img.flags.writeable:
            img = img.copy()
        img_blurred_full = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        img[mask > 0] = img_blurred_full[mask > 0]


def _yolo_boxes_from_result(result):
    """Return list of xyxy ndarray for a single YOLO result."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    # YOLO results usually have .cpu() and .numpy() if they are tensors
    return [b.xyxy[0].cpu().numpy() for b in result.boxes]

def blur_equirectangular_images(
    source_dir: str,
    base_output_dir: str,
    face_weights_path: str,
    plate_weights_path: str,
    sam_checkpoint_path: str,
    sam_hf_model_name: str,
    device: str = "cuda",
    yolo_confidence_threshold: float = 0.25,
    min_box_size_px: int = 15,
    blur_kernel_size: int = 31
):
    """
    Processes all JPG images found in source_dir (and its subdirectories) for anonymization.
    """
    blurred_images_dir = os.path.join(base_output_dir, "02_blurred_images")
    ensure_dir_exists(blurred_images_dir)
    log_json_path = os.path.join(blurred_images_dir, "blur_log.json")

    try:
        face_detector = YOLO(face_weights_path)
        plate_detector = YOLO(plate_weights_path)
        sam_predictor = None
        if SAM2ImagePredictor and sam_checkpoint_path and sam_hf_model_name:
             if os.path.exists(sam_checkpoint_path):
                sam_predictor = SAM2ImagePredictor.from_pretrained(
                    sam_hf_model_name, checkpoint=sam_checkpoint_path, device=device
                )
                print("SAM predictor loaded.")
             else:
                print(f"Warning: SAM checkpoint not found at {sam_checkpoint_path}. Proceeding without SAM.")
        else:
            print("SAM not configured or available. Proceeding without SAM refinement.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

    log_entries = []
    
    image_files_to_process = []
    print(f"Searching for JPG/JPEG images in and under: {source_dir}")
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg")):
                image_files_to_process.append(os.path.join(root, filename))
    
    image_files_to_process.sort()

    if not image_files_to_process:
        print(f"No JPG/JPEG images found in or under {source_dir}. Skipping blurring.")
        with open(log_json_path, "w", encoding="utf-8") as fp:
            json.dump(log_entries, fp, indent=2)
        return blurred_images_dir, log_json_path

    print(f"Starting blurring process for {len(image_files_to_process)} images...")
    for image_full_path in tqdm(image_files_to_process, desc="Blurring Images"):
        try:
            equi_bgr = cv2.imread(image_full_path)
            if equi_bgr is None:
                print(f"Warning: Could not read image {image_full_path}. Skipping.")
                continue
            equi_rgb = cv2.cvtColor(equi_bgr, cv2.COLOR_BGR2RGB)

            cube_face_width = equi_rgb.shape[1] // 4
            cube_faces_chw = equi2cube(
                equi=equi_rgb.transpose(2, 0, 1),
                rots={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                w_face=cube_face_width,
                cube_format="list"
            )
            
            processed_cube_faces_chw = [] 

            for face_idx, face_chw in enumerate(cube_faces_chw):
                face_hwc = face_chw.transpose(1, 2, 0).astype(np.uint8).copy()
                face_resized_for_yolo = cv2.resize(face_hwc, (640, 640))
                current_face_combined_mask = np.zeros((face_hwc.shape[0], face_hwc.shape[1]), dtype=np.uint8)

                face_detections = face_detector.predict(face_resized_for_yolo, conf=yolo_confidence_threshold, imgsz=640, verbose=False)[0]
                plate_detections = plate_detector.predict(face_resized_for_yolo, conf=yolo_confidence_threshold, imgsz=640, verbose=False)[0]
                yolo_det_boxes = _yolo_boxes_from_result(face_detections) + _yolo_boxes_from_result(plate_detections)

                for box_xyxy_yolo_coords in yolo_det_boxes:
                    scale_x = face_hwc.shape[1] / 640.0
                    scale_y = face_hwc.shape[0] / 640.0
                    x1, y1 = int(box_xyxy_yolo_coords[0] * scale_x), int(box_xyxy_yolo_coords[1] * scale_y)
                    x2, y2 = int(box_xyxy_yolo_coords[2] * scale_x), int(box_xyxy_yolo_coords[3] * scale_y)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(face_hwc.shape[1], x2), min(face_hwc.shape[0], y2)
                    box_orig_face_coords = [x1, y1, x2, y2]

                    if (box_orig_face_coords[2] - box_orig_face_coords[0] < min_box_size_px) or \
                       (box_orig_face_coords[3] - box_orig_face_coords[1] < min_box_size_px):
                        continue

                    if sam_predictor:
                        try:
                            sam_predictor.set_image(face_hwc)
                            with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
                                sam_masks_out, _, _ = sam_predictor.predict(
                                    box=np.array(box_orig_face_coords), 
                                    multimask_output=False
                                )
                            # ---- THIS IS THE CORRECTED LINE ----
                            # sam_masks_out[0] is likely already a NumPy array if this error occurred.
                            # No .cpu().numpy() needed if it's already a NumPy array.
                            current_face_combined_mask[sam_masks_out[0].astype(bool)] = 255
                            # ---- END CORRECTION ----
                        except Exception as e_sam:
                            print(f"Warning: SAM prediction failed for a box in {image_full_path}, face {face_idx}: {e_sam}. Using YOLO box for blurring.")
                            current_face_combined_mask[y1:y2, x1:x2] = 255 # Fallback
                    else: 
                        current_face_combined_mask[y1:y2, x1:x2] = 255
                
                _gaussian_blur_masked_area(face_hwc, current_face_combined_mask, kernel_size=blur_kernel_size)
                processed_cube_faces_chw.append(face_hwc.transpose(2,0,1)) 

            equi_blurred_chw = cube2equi(
                processed_cube_faces_chw, 
                cube_format="list",
                height=equi_rgb.shape[0],
                width=equi_rgb.shape[1],
                clip_output=True,
                mode="bilinear"
            )
            
            equi_blurred_hwc_rgb = equi_blurred_chw.transpose(1, 2, 0)
            if not np.issubdtype(equi_blurred_hwc_rgb.dtype, np.uint8):
                equi_blurred_hwc_rgb = np.clip(equi_blurred_hwc_rgb, 0, 255).astype(np.uint8)
            equi_blurred_hwc_bgr = cv2.cvtColor(equi_blurred_hwc_rgb, cv2.COLOR_RGB2BGR)
            
            original_basename = os.path.basename(image_full_path)
            output_filename_base = original_basename.rsplit(".", 1)[0]
            output_filename = f"{output_filename_base}_blurred.jpg"
            
            output_path = os.path.join(blurred_images_dir, output_filename)
            cv2.imwrite(output_path, equi_blurred_hwc_bgr)
            log_entries.append({"source_file": image_full_path, "blurred_file": output_path, "status": "success"})

        except Exception as e:
            print(f"Error processing {image_full_path}: {e}")
            log_entries.append({"source_file": image_full_path, "blurred_file": None, "status": "error", "detail": str(e)})

    with open(log_json_path, "w", encoding="utf-8") as fp:
        json.dump(log_entries, fp, indent=2)

    print(f"✅ Blurring complete. Redacted panoramas saved to → {blurred_images_dir}")
    print(f"📝 Blurring log written to → {log_json_path}")
    return blurred_images_dir, log_json_path