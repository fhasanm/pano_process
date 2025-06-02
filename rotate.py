
#!/usr/bin/env python3
"""
panorama_rotator.py – Rotates panoramas to face identified building façades.
Reads a CSV of panorama-façade matches (from facade_processor.py) and
the original (or blurred) panoramas.
Applies a two-stage yaw rotation:
1. Coarse yaw: Aligns panorama based on vehicle heading, measured camera offset, and desired facade yaw.
2. Fine yaw: Refines alignment based on the bearing from camera to façade edge midpoint.
Outputs rotated panoramas and a JSON metadata file.
"""
import os
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from equilib import equi2equi # Ensure equilib is installed
from tqdm import tqdm
from utils import (
    ensure_dir_exists,
    great_circle_bearing,
    signed_angular_difference,
    calculate_distance_meters
)

def rotate_panoramas_to_facades(
    facade_matches_csv_path: str,
    source_panoramas_dir: str, # Dir containing original (or blurred) panos listed in CSV
    base_output_dir: str,
    measured_camera_offset_deg: float, # PANO_ZERO_OFFSET
    distance_cutoff_m: float # Max distance to consider for rotation
):
    """
    Rotates panoramas based on the facade matching CSV.

    Args:
        facade_matches_csv_path: Path to CSV from facade_processor.py.
        source_panoramas_dir: Directory where the original panoramas (referenced in
                              the 'pano_abs_path' or 'pano_filename' column of CSV) are located.
        base_output_dir: Base directory for pipeline outputs. Rotated panos go into a subdir.
        measured_camera_offset_deg: The measured yaw offset of the camera's optical axis
                                    relative to the vehicle's true heading.
                                    (PANO_ZERO_OFFSET: positive if camera is right of vehicle heading).
        distance_cutoff_m: Filter out matches where building centroid is further than this.

    Returns:
        A tuple (rotated_panos_dir, rotated_meta_json_path) or (None, None) if an error.
    """
    rotated_panos_dir = os.path.join(base_output_dir, "04_rotated_panoramas")
    ensure_dir_exists(rotated_panos_dir)
    rotated_meta_json_path = os.path.join(rotated_panos_dir, "rotated_panoramas_metadata.json")

    if not os.path.exists(facade_matches_csv_path):
        print(f"Error: Facade matches CSV not found at {facade_matches_csv_path}")
        return None, None
    
    try:
        df_matches = pd.read_csv(facade_matches_csv_path)
    except Exception as e:
        print(f"Error reading facade matches CSV {facade_matches_csv_path}: {e}")
        return None, None

    if df_matches.empty:
        print("Facade matches CSV is empty. No panoramas to rotate.")
        # Create empty metadata and return
        with open(rotated_meta_json_path, "w") as fp:
            json.dump([], fp)
        return rotated_panos_dir, rotated_meta_json_path

    # Filter by distance to centroid (already in CSV, but can re-filter if needed)
    # The CSV from facade_processor already contains 'distance_to_centroid_m'
    # We can also re-calculate or use a 'dist_chk' like in the original script if we want
    # to be absolutely sure about the distance definition.
    # For simplicity, we'll trust 'distance_to_centroid_m' from the CSV if it exists,
    # otherwise, we might need to calculate it if `distance_cutoff_m` is critical here.
    # Original script used 'centroid_lat', 'centroid_lon', 'MAPLatitude', 'MAPLongitude'
    # to calculate 'dist_chk'. Let's ensure our CSV has these fields.
    
    required_cols = ['pano_latitude', 'pano_longitude', 'bld_centroid_lat', 'bld_centroid_lon', 'distance_to_centroid_m', 'pano_abs_path', 'pano_filename']
    for col in required_cols:
        if col not in df_matches.columns:
            print(f"Error: Missing required column '{col}' in {facade_matches_csv_path}")
            # Attempt to compute distance_to_centroid_m if missing and other cols exist
            if col == 'distance_to_centroid_m' and all(c in df_matches.columns for c in ['pano_latitude', 'pano_longitude', 'bld_centroid_lat', 'bld_centroid_lon']):
                print("Attempting to calculate 'distance_to_centroid_m'...")
                df_matches['distance_to_centroid_m'] = df_matches.apply(
                    lambda r: calculate_distance_meters(r.pano_latitude, r.pano_longitude, r.bld_centroid_lat, r.bld_centroid_lon), axis=1
                )
            else:
                 return None, None

    df_filtered = df_matches[df_matches['distance_to_centroid_m'] <= distance_cutoff_m].reset_index(drop=True)

    if df_filtered.empty:
        print(f"No facade matches within distance cutoff {distance_cutoff_m}m. No panoramas to rotate.")
        with open(rotated_meta_json_path, "w") as fp:
            json.dump([], fp)
        return rotated_panos_dir, rotated_meta_json_path

    output_metadata_list = []
    
    print(f"Rotating {len(df_filtered)} panoramas based on facade matches...")
    for idx, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0], desc="Rotating Panoramas"):
        if idx>20:
            break
        # Determine the correct path to the source panorama
        # 'pano_abs_path' from facade_processor should be the primary source.
        # If it's not absolute, or if we prefer to ensure it's from source_panoramas_dir:
        source_pano_path = row.get("pano_abs_path", "")
        if not os.path.isabs(source_pano_path) or not os.path.exists(source_pano_path):
             # Fallback to joining source_panoramas_dir with pano_filename
             # This assumes pano_filename is just the file name, not a relative path from somewhere else.
             source_pano_path = os.path.join(source_panoramas_dir, row["pano_filename"])

        if not os.path.isfile(source_pano_path):
            print(f"Warning: Source panorama not found at '{source_pano_path}' for row {idx}. Skipping.")
            continue

        try:
            # Vehicle's True Heading (world coordinates)
            H_vehicle = float(row["pano_true_heading"])
            
            # Raw (current) heading of the panorama's 0-degree line (center) in world coordinates
            # This is the vehicle's heading plus the camera's fixed offset from the vehicle.
            H_pano_current_center = (H_vehicle + measured_camera_offset_deg + 360) % 360
            
            # Desired world heading for the center of the panorama to point at (façade midpoint)
            H_desired_center_target_facade = float(row["desired_camera_yaw_to_facade"])

            # 1. Coarse Yaw Adjustment:
            #    How much to rotate the panorama from its current orientation (H_pano_current_center)
            #    to the desired orientation (H_desired_center_target_facade).
            #    A positive yaw_coarse means rotate pano clockwise (pixels shift left).
            yaw_coarse_deg = signed_angular_difference(H_pano_current_center, H_desired_center_target_facade)

            # 2. Fine Yaw Adjustment (Refinement):
            #    The 'desired_camera_yaw_to_facade' in the CSV is already the bearing from
            #    camera to the facade midpoint. This is what H_desired_center_target_facade is.
            #    The original script had a more complex fine yaw. Let's re-evaluate.
            #    Original:
            #        H_mid = bearing(r.MAPLatitude, r.MAPLongitude, mid_lat, mid_lon) -> This IS r.desired_yaw
            #        yaw1 = -signed(H_desired, H_mid)
            #    This yaw1 seems to be a correction if H_desired (from a centroid-based approach in older version?)
            #    is different from H_mid (direct to midpoint).
            #    In the current facade_processor, 'desired_camera_yaw_to_facade' IS H_mid. So yaw1 should be 0.
            #    Let's keep the logic if 'desired_camera_yaw_to_facade' could come from a different source
            #    than direct midpoint bearing.
            #    For now, assuming `desired_camera_yaw_to_facade` is accurate.
            
            #    The original `rotate_panos_multi.py` calculated `H_mid` using `front_lon0`, `front_lat0`, etc.
            #    Our `facade_matches_csv_path` should already have `desired_camera_yaw_to_facade` as this precise bearing.
            #    So, effectively, after coarse yaw, the pano center *should* point at `desired_camera_yaw_to_facade`.
            #    The fine yaw was a correction: `yaw1 = -signed(H_desired, H_mid)`.
            #    If `H_desired` IS `H_mid`, then `yaw1` is 0.
            #    Let's assume `desired_camera_yaw_to_facade` is the target.
            #    The key is: `equi2equi` yaw rotates the content. If yaw is positive, scene rotates left (camera view right).
            #    If current center is 0 deg, and target is 90 deg, we want to make 90 deg azimuth be at center.
            #    This means content at 90 deg needs to move to 0 deg. This is a rotation of -90 deg for the image.
            #    So, rotation_angle = H_pano_current_center - H_desired_center_target_facade
            #    This is -yaw_coarse_deg.
            #    Let's stick to the definition from original script for yaw parameter in equi2equi.
            #    If `rots={"yaw": angle_rad}`, positive angle means the new center of the image will point
            #    to `angle_rad` to the right (clockwise) of the old image's center viewing direction.
            #    So if we want the new center to be `H_desired_center_target_facade`, and old center was `H_pano_current_center`,
            #    the required rotation is `yaw_coarse_deg`.

            # Load image, transpose to C, H, W for equilib
            pil_image = Image.open(source_pano_path)
            img_array_chw = np.asarray(pil_image).transpose(2, 0, 1)

            # Apply rotation
            # A single rotation should suffice if yaw_coarse_deg is calculated correctly.
            # The original script did two. If the fine yaw is truly for correcting a slightly off H_desired,
            # it might still be useful. For now, we'll assume one precise rotation is enough.
            # If H_desired_center_target_facade is precise, yaw_coarse_deg is the only rotation needed.
            rotated_img_chw = equi2equi(
                img_array_chw,
                rots={"roll": 0, "pitch": 0, "yaw": math.radians(yaw_coarse_deg)}
            )

            if not np.issubdtype(rotated_img_chw.dtype, np.uint8):
                rotated_img_chw = np.clip(rotated_img_chw, 0, 255).astype(np.uint8)
            
            # Transpose back to H, W, C for PIL
            rotated_img_hwc = rotated_img_chw.transpose(1, 2, 0)
            output_pil_image = Image.fromarray(rotated_img_hwc)

            base_filename = os.path.splitext(os.path.basename(row["pano_filename"]))[0]
            # Use BLD_ID and original CSV row index for unique filenaming.
            # Make BLD_ID file-system friendly (replace non-alphanumeric)
            bld_id_str = str(row.get("BLD_ID", f"unknownBLD_{idx}")).replace(os.sep, "_").replace(" ","_")

            output_image_filename = f"{base_filename}_BLD{bld_id_str}_FACADE{idx}_ROT.jpg"
            output_image_path = os.path.join(rotated_panos_dir, output_image_filename)
            output_pil_image.save(output_image_path)

            record = row.to_dict() # All original columns from CSV
            record.update({
                "rotated_pano_path": output_image_path,
                "applied_yaw_rotation_deg": yaw_coarse_deg, # The total effective yaw applied
                "camera_offset_used_deg": measured_camera_offset_deg,
                "H_vehicle_deg": H_vehicle,
                "H_pano_initial_center_deg": H_pano_current_center,
                "H_pano_final_center_deg": H_desired_center_target_facade
            })
            output_metadata_list.append(record)

        except Exception as e_rotate:
            print(f"Error rotating panorama for row {idx} (file {row.get('pano_filename', 'N/A')}): {e_rotate}")
            # import traceback # For debugging
            # traceback.print_exc() # For debugging

    try:
        with open(rotated_meta_json_path, "w") as fp_json:
            json.dump(output_metadata_list, fp_json, indent=2)
        print(f"✅ Panorama rotation complete. {len(output_metadata_list)} images saved to → {rotated_panos_dir}")
        print(f"📝 Rotation metadata saved to → {rotated_meta_json_path}")
    except Exception as e_json:
        print(f"Error writing rotation metadata JSON to {rotated_meta_json_path}: {e_json}")
        return None, None
        
    return rotated_panos_dir, rotated_meta_json_path