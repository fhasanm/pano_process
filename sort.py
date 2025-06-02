
#!/usr/bin/env python3
"""
sort.py – Sorts extracted cube faces into per-building folders.
Reads metadata from cube_extractor.py and, for each building ID (BLD_ID),
creates a subdirectory. Copies/moves the relevant cube faces into these
subdirectories. Also creates a 'building_info.json' in each building's
folder, containing consolidated metadata about the building, including its
geographic coordinates (centroid) and damage assessment information (if available).
"""
import os
import json
import shutil
import csv
from collections import defaultdict
from tqdm import tqdm
from utils import ensure_dir_exists

def sort_cube_faces_by_building(
    cube_faces_metadata_json_path: str, # From cube_extractor.py
    base_output_dir: str,
    building_damage_csv_path: str, # CSV with BLD_ID and DAMAGE columns
    move_files: bool = False
):
    """
    Sorts cube faces (typically front faces) into folders per building ID.
    Also adds building-specific information, including geographic lat/lon and damage.

    Args:
        cube_faces_metadata_json_path: Path to the JSON metadata from cube_extractor.py.
                                       This file lists cube faces and associated building info.
        base_output_dir: Base directory for pipeline outputs. Sorted outputs go into a subdir.
        building_damage_csv_path: Path to a CSV file that contains at least 'BLD_ID'
                                  and 'DAMAGE' columns for buildings.
        move_files: If True, moves cube face files. If False (default), copies them.

    Returns:
        Path to the main sorted output directory, or None if an error occurs.
    """
    sorted_output_root_dir = os.path.join(base_output_dir, "06_sorted_outputs_by_building")
    ensure_dir_exists(sorted_output_root_dir)

    if not os.path.exists(cube_faces_metadata_json_path):
        print(f"Error: Cube faces metadata JSON not found at {cube_faces_metadata_json_path}")
        return None
    if not os.path.exists(building_damage_csv_path):
        print(f"Warning: Building damage CSV not found at {building_damage_csv_path}. DAMAGE field will be empty.")
        damage_map = {}
    else:
        # --- 1. Load building damage information from CSV ---
        damage_map = {}
        try:
            with open(building_damage_csv_path, mode='r', encoding='utf-8', newline='') as f_csv:
                reader = csv.DictReader(f_csv)
                if "BLD_ID" not in reader.fieldnames or "DAMAGE" not in reader.fieldnames:
                    print(f"Warning: CSV {building_damage_csv_path} must contain 'BLD_ID' and 'DAMAGE' columns. DAMAGE will be empty.")
                else:
                    for row in reader:
                        bld_id_csv = str(row.get("BLD_ID", "")).strip()
                        if bld_id_csv:
                            damage_map[bld_id_csv] = row.get("DAMAGE", "").strip()
            print(f"Loaded damage information for {len(damage_map)} buildings from {building_damage_csv_path}")
        except Exception as e_csv:
            print(f"Error reading damage CSV {building_damage_csv_path}: {e_csv}. DAMAGE field will be empty.")
            damage_map = {}


    # --- 2. Load cube faces metadata ---
    try:
        with open(cube_faces_metadata_json_path, 'r') as f_json:
            all_cube_face_records = json.load(f_json)
    except Exception as e_json:
        print(f"Error reading cube faces metadata {cube_faces_metadata_json_path}: {e_json}")
        return None

    if not all_cube_face_records:
        print("No cube face records found to sort.")
        return sorted_output_root_dir # Return dir path even if nothing to sort

    # --- 3. Group records by BLD_ID ---
    # Each record in all_cube_face_records corresponds to one original panorama's processing,
    # which resulted in one or more cube faces for a specific BLD_ID.
    records_grouped_by_bld_id = defaultdict(list)
    for record in all_cube_face_records:
        bld_id = str(record.get("BLD_ID", "UnknownBuilding")).strip()
        records_grouped_by_bld_id[bld_id].append(record)
    
    print(f"Sorting {len(all_cube_face_records)} cube face sets into {len(records_grouped_by_bld_id)} building folders...")

    # --- 4. Process each building ---
    # Define which fields from the input records should be part of the building_info.json
    # These are typically building-level attributes. We take them from the first record for that building.
    # Geographic lat/lon for the building centroid should be 'bld_centroid_lat' and 'bld_centroid_lon'.
    building_info_fields_to_extract = [
        "BLD_ID", "bld_height", "bld_elevation", "bld_source_dataset", 
        "bld_capture_date", "bld_status",
        "bld_centroid_lon", "bld_centroid_lat", # Geographic coordinates of the building
        # Add any other relevant building-specific fields present in the records
        # e.g. if facade_processor added building AREA, Shape__Area etc. from geojson
        # "bld_area", "bld_perimeter" 
    ]


    for bld_id, building_records in tqdm(records_grouped_by_bld_id.items(), desc="Sorting by Building"):
        # Sanitize BLD_ID for directory naming (replace slashes, spaces, etc.)
        safe_bld_id_dirname = bld_id.replace(os.sep, "_").replace(" ", "_").replace(":", "_")
        current_building_output_dir = os.path.join(sorted_output_root_dir, safe_bld_id_dirname)
        ensure_dir_exists(current_building_output_dir)

        # --- Create building_info.json ---
        # Use the first record for this building to populate general building info
        first_record_for_building = building_records[0]
        building_info_data = {}
        for field in building_info_fields_to_extract:
            building_info_data[field] = first_record_for_building.get(field)
        
        # Add DAMAGE information
        building_info_data["DAMAGE_assessment"] = damage_map.get(bld_id, "N/A") # Use bld_id as key

        # Add a list of all views (cube face sets) associated with this building
        building_info_data["associated_views"] = []

        # --- Copy/Move cube face files and collect view-specific metadata ---
        for view_record in building_records:
            extracted_faces_dict = view_record.get("extracted_cube_faces", {})
            if not extracted_faces_dict:
                continue # No faces were extracted for this particular view record

            view_specific_info = {
                "original_pano_filename": view_record.get("pano_filename"),
                "original_pano_abs_path": view_record.get("pano_abs_path"), # from facade_processor
                "rotated_pano_path": view_record.get("rotated_pano_path"), # from panorama_rotator
                "view_cube_faces": {}, # Store paths to faces copied/moved into this bld_dir
                # Include other view-specific details if needed:
                "pano_latitude": view_record.get("pano_latitude"),
                "pano_longitude": view_record.get("pano_longitude"),
                "pano_true_heading": view_record.get("pano_true_heading"),
                "distance_to_centroid_m": view_record.get("distance_to_centroid_m"),
                "desired_camera_yaw_to_facade": view_record.get("desired_camera_yaw_to_facade"),
                "applied_yaw_rotation_deg": view_record.get("applied_yaw_rotation_deg"),
            }

            for face_name, source_face_path in extracted_faces_dict.items():
                if os.path.isfile(source_face_path):
                    destination_face_path = os.path.join(current_building_output_dir, os.path.basename(source_face_path))
                    try:
                        if move_files:
                            shutil.move(source_face_path, destination_face_path)
                        else:
                            shutil.copy2(source_face_path, destination_face_path)
                        view_specific_info["view_cube_faces"][face_name] = destination_face_path
                    except Exception as e_file_op:
                        print(f"Error {'moving' if move_files else 'copying'} file {source_face_path} to {destination_face_path}: {e_file_op}")
                else:
                    print(f"Warning: Cube face file {source_face_path} not found for BLD_ID {bld_id}, face {face_name}. Skipping.")
            
            building_info_data["associated_views"].append(view_specific_info)

        # Save the consolidated building_info.json for this building
        building_info_json_path = os.path.join(current_building_output_dir, "building_info.json")
        try:
            with open(building_info_json_path, "w") as fp_bld_json:
                json.dump(building_info_data, fp_bld_json, indent=2)
        except Exception as e_bld_json:
            print(f"Error writing building_info.json for BLD_ID {bld_id}: {e_bld_json}")

    print(f"✅ Output sorting complete. Sorted outputs are in → {sorted_output_root_dir}")
    return sorted_output_root_dir