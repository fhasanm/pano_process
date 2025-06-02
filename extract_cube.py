
#!/usr/bin/env python3
"""
cube_extractor.py – Extracts selected cubemap faces from rotated equirectangular panoramas.
Reads metadata about rotated panoramas, converts each to a cubemap,
and saves specified faces (e.g., front, right, etc.) as PNG images.
Outputs the paths to these cube faces along with consolidated metadata in a JSON file.
"""
import os
import json
import numpy as np
from PIL import Image
from equilib import equi2cube # Ensure equilib is installed
from tqdm import tqdm
from utils import ensure_dir_exists
from config import ALL_POSSIBLE_CUBE_FACES # Using the list from config

def extract_cubemap_faces(
    rotated_panoramas_meta_json_path: str,
    # rotated_panoramas_base_dir: str, # Path in JSON should be absolute or resolvable
    base_output_dir: str,
    faces_to_extract: list # List of face names like ["front", "left"]
):
    """
    Converts rotated panoramas to cubemaps and saves selected faces.

    Args:
        rotated_panoramas_meta_json_path: Path to JSON metadata from panorama_rotator.py.
                                          This JSON contains paths to the rotated panoramas.
        base_output_dir: Base directory for pipeline outputs. Cube faces go into a subdir.
        faces_to_extract: A list of strings specifying which cube faces to save.
                          Valid names are: "front", "right", "back", "left", "top", "bottom".
                          If an empty list is provided, all faces will be extracted.

    Returns:
        A tuple (cube_faces_output_dir, cube_faces_meta_json_path) or (None, None) if an error.
    """
    cube_faces_output_dir = os.path.join(base_output_dir, "05_cube_faces")
    ensure_dir_exists(cube_faces_output_dir)
    cube_faces_meta_json_path = os.path.join(cube_faces_output_dir, "cube_faces_metadata.json")

    if not faces_to_extract: # If empty, extract all
        faces_to_extract = ALL_POSSIBLE_CUBE_FACES
    else: # Validate user input against all possible faces
        for face_name in faces_to_extract:
            if face_name not in ALL_POSSIBLE_CUBE_FACES:
                print(f"Error: Invalid face name '{face_name}' in faces_to_extract. Valid names are: {ALL_POSSIBLE_CUBE_FACES}")
                return None, None
    
    if not os.path.exists(rotated_panoramas_meta_json_path):
        print(f"Error: Rotated panoramas metadata JSON not found at {rotated_panoramas_meta_json_path}")
        return None, None

    try:
        with open(rotated_panoramas_meta_json_path, 'r') as f:
            rotated_pano_records = json.load(f)
    except Exception as e:
        print(f"Error reading rotated panoramas metadata: {e}")
        return None, None

    if not rotated_pano_records:
        print("No rotated panorama records found. Skipping cube face extraction.")
        with open(cube_faces_meta_json_path, "w") as fp: json.dump([], fp) # Write empty list
        return cube_faces_output_dir, cube_faces_meta_json_path

    output_metadata_records = []
    
    print(f"Extracting {len(faces_to_extract)} cube face(s) for {len(rotated_pano_records)} rotated panoramas...")
    for record in tqdm(rotated_pano_records, desc="Extracting Cube Faces"):
        rotated_pano_path = record.get("rotated_pano_path")
        if not rotated_pano_path or not os.path.isfile(rotated_pano_path):
            print(f"Warning: Rotated panorama path '{rotated_pano_path}' not found or invalid in record. Skipping.")
            continue

        try:
            pil_image = Image.open(rotated_pano_path)
            # Convert to NumPy array HWC, then CHW for equi2cube
            img_array_hwc = np.asarray(pil_image)
            img_array_chw = np.transpose(img_array_hwc, (2, 0, 1))

            # Determine face width (standard 90-degree FOV for each cube face)
            # Equirectangular width is 4 times the cube face width
            cube_face_width = img_array_hwc.shape[1] // 4

            # Convert equirectangular to a list of cube faces (CHW format)
            # No additional rotation needed here as panoramas are already rotated.
            # equi2cube returns faces in order: [+z (front), +x (right), -z (back), -x (left), +y (top), -y (bottom)]
            # when z_down=False (default).
            list_of_cube_faces_chw = equi2cube(
                equi=img_array_chw,
                rots={"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, # No rotation
                w_face=cube_face_width,
                cube_format='list', # Returns a list of numpy arrays
                z_down=False,       # Standard orientation: +Y up, +Z front
                clip_output=True,
                mode='bilinear'     # Interpolation mode
            )

            saved_face_paths = {}
            base_filename_no_ext = os.path.splitext(os.path.basename(rotated_pano_path))[0]

            for i, face_chw_data in enumerate(list_of_cube_faces_chw):
                current_face_name = ALL_POSSIBLE_CUBE_FACES[i] # Get name based on standard order

                if current_face_name in faces_to_extract:
                    # Ensure data is uint8 and convert CHW to HWC for saving with PIL
                    if not np.issubdtype(face_chw_data.dtype, np.uint8):
                        face_chw_data = np.clip(face_chw_data, 0, 255).astype(np.uint8)
                    
                    face_hwc_data = np.transpose(face_chw_data, (1, 2, 0))
                    face_image_pil = Image.fromarray(face_hwc_data)
                    
                    # Define output path for this cube face
                    face_output_filename = f"{base_filename_no_ext}_{current_face_name}.png" # Save as PNG
                    face_output_path = os.path.join(cube_faces_output_dir, face_output_filename)
                    
                    face_image_pil.save(face_output_path)
                    saved_face_paths[current_face_name] = face_output_path
            
            # Update the original record with paths to the extracted cube faces
            # Make a copy to avoid modifying the list being iterated over if it's the same object
            updated_record = dict(record) 
            updated_record["extracted_cube_faces"] = saved_face_paths
            output_metadata_records.append(updated_record)

        except Exception as e_extract:
            print(f"Error extracting cube faces for {rotated_pano_path}: {e_extract}")
            # import traceback # For debugging
            # traceback.print_exc() # For debugging

    try:
        with open(cube_faces_meta_json_path, "w") as fp_json:
            json.dump(output_metadata_records, fp_json, indent=2)
        print(f"✅ Cube face extraction complete. Images saved in → {cube_faces_output_dir}")
        print(f"📝 Cube faces metadata saved to → {cube_faces_meta_json_path}")
    except Exception as e_json:
        print(f"Error writing cube faces metadata JSON: {e_json}")
        return None, None
        
    return cube_faces_output_dir, cube_faces_meta_json_path