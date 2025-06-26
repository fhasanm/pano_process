#!/usr/bin/env python3
"""
extract_cube.py – Extracts selected cubemap faces from rotated equirectangular panoramas.
Reads metadata about rotated panoramas, converts each to a cubemap,
and saves specified faces (e.g., front, right, etc.) as PNG images.
Outputs the paths to these cube faces along with consolidated metadata in a JSON file.
"""
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from equilib import equi2cube # Ensure equilib is installed
from tqdm import tqdm
from utils import ensure_dir_exists
from config import ALL_POSSIBLE_CUBE_FACES # Using the list from config

def extract_cubemap_faces(
    rotated_panoramas_meta_json_path: str,
    base_output_dir: str,
    faces_to_extract: list # List of face names like ["front", "left"]
):
    """
    Converts rotated panoramas to cubemaps and saves selected faces.

    Args:
        rotated_panoramas_meta_json_path (str): Path to JSON metadata from the rotation stage.
        base_output_dir (str): Base directory for the current video's pipeline outputs.
        faces_to_extract (list): A list of strings specifying which cube faces to save.

    Returns:
        A tuple (path_to_images_directory, path_to_metadata_json) or (None, None).
    """
    # --- Setup paths using pathlib for robustness ---
    video_output_dir = Path(base_output_dir)
    extraction_output_dir = video_output_dir / "05_cube_faces"
    cubeface_images_dir = extraction_output_dir / "cubeface_image_files"
    ensure_dir_exists(cubeface_images_dir)
    
    output_meta_path = extraction_output_dir / "cube_faces_metadata.json"
    
    meta_json_path = Path(rotated_panoramas_meta_json_path)

    if not faces_to_extract:
        faces_to_extract = ALL_POSSIBLE_CUBE_FACES
        
    if not meta_json_path.is_file():
        print(f"Error: Rotated panoramas metadata JSON not found at {meta_json_path}")
        return None, None

    try:
        with open(meta_json_path, 'r') as f:
            rotated_pano_records = json.load(f)
    except Exception as e:
        print(f"Error reading rotated panoramas metadata: {e}")
        return None, None

    if not rotated_pano_records:
        print("No rotated panorama records found. Skipping cube face extraction.")
        with open(output_meta_path, "w") as fp: json.dump([], fp)
        return str(cubeface_images_dir), str(output_meta_path)

    output_metadata_records = []
    
    # The source for rotated images is relative to the metadata JSON we're reading
    rotated_images_source_dir = meta_json_path.parent

    print(f"Extracting {len(faces_to_extract)} cube face(s) for {len(rotated_pano_records)} rotated panoramas...")
    for record in tqdm(rotated_pano_records, desc="Extracting Cube Faces"):
        # The key in the JSON should point to the filename of the rotated image
        rotated_image_filename = record.get("rotated_image_path_relative") 
        if not rotated_image_filename:
            # Fallback for compatibility with older 'rotate.py' versions
            rotated_image_filename = Path(record.get("rotated_pano_path", "")).name
        
        if not rotated_image_filename:
            print(f"Warning: Could not find rotated panorama path in record. Skipping.")
            continue
            
        rotated_pano_path = rotated_images_source_dir / rotated_image_filename
        
        if not rotated_pano_path.is_file():
            print(f"Warning: Rotated panorama file '{rotated_pano_path}' not found. Skipping.")
            continue

        try:
            pil_image = Image.open(rotated_pano_path)
            img_array_hwc = np.asarray(pil_image)
            # Handle grayscale or RGBA images
            if img_array_hwc.ndim == 2:
                img_array_hwc = np.stack([img_array_hwc]*3, axis=-1)
            if img_array_hwc.shape[2] == 4:
                img_array_hwc = img_array_hwc[..., :3]

            img_array_chw = np.transpose(img_array_hwc, (2, 0, 1))
            cube_face_width = img_array_hwc.shape[1] // 4

            list_of_cube_faces_chw = equi2cube(
                equi=img_array_chw,
                rots={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                w_face=cube_face_width,
                cube_format='list'
            )

            saved_face_paths = {}
            base_filename_no_ext = rotated_pano_path.stem

            for i, face_chw_data in enumerate(list_of_cube_faces_chw):
                current_face_name = ALL_POSSIBLE_CUBE_FACES[i]

                if current_face_name in faces_to_extract:
                    face_hwc_data = np.transpose(np.clip(face_chw_data, 0, 255).astype(np.uint8), (1, 2, 0))
                    face_image_pil = Image.fromarray(face_hwc_data)
                    
                    face_output_filename = f"{base_filename_no_ext}_{current_face_name}.png"
                    face_output_path = cubeface_images_dir / face_output_filename
                    
                    face_image_pil.save(face_output_path)
                    
                    # <<< THE CRITICAL FIX >>>
                    # Store only the filename, not the full path, in the metadata.
                    saved_face_paths[current_face_name] = face_output_filename
            
            updated_record = dict(record) 
            updated_record["extracted_cube_faces"] = saved_face_paths
            output_metadata_records.append(updated_record)

        except Exception as e_extract:
            print(f"Error extracting cube faces for {rotated_pano_path}: {e_extract}")

    try:
        with open(output_meta_path, "w") as fp_json:
            json.dump(output_metadata_records, fp_json, indent=2)
        print(f"✅ Cube face extraction complete. Images saved in → {cubeface_images_dir}")
        print(f"📝 Cube faces metadata saved to → {output_meta_path}")
    except Exception as e_json:
        print(f"Error writing cube faces metadata JSON: {e_json}")
        return None, None
        
    return str(cubeface_images_dir), str(output_meta_path)
