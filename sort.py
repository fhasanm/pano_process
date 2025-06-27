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
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir_exists

def sort_cube_faces_by_building(
    cube_faces_metadata_json_path: str, # From cube_extractor.py
    building_damage_csv_path: str,      # CSV with BLD_ID and DAMAGE columns
    final_sorted_dir: str,              # The specific, consolidated directory to save sorted outputs into.
    move_files: bool = False
):
    """
    Sorts cube faces into folders per building ID.

    Args:
        cube_faces_metadata_json_path (str): Path to the JSON metadata from cube_extractor.py.
        building_damage_csv_path (str): Path to a CSV file with building damage info.
        final_sorted_dir (str): The specific directory to save sorted outputs into.
        move_files (bool): If True, moves cube face files. If False (default), copies them.
    """
    sorted_output_root_dir = Path(final_sorted_dir)
    ensure_dir_exists(sorted_output_root_dir)

    meta_json_path_obj = Path(cube_faces_metadata_json_path)
    if not meta_json_path_obj.is_file():
        print(f"Error: Cube faces metadata JSON not found at {meta_json_path_obj}")
        return None

    # --- 1. Load building damage information from CSV ---
    damage_map = {}
    if Path(building_damage_csv_path).is_file():
        try:
            with open(building_damage_csv_path, mode='r', encoding='utf-8', newline='') as f_csv:
                reader = csv.DictReader(f_csv)
                if "BLD_ID" in reader.fieldnames and "DAMAGE" in reader.fieldnames:
                    for row in reader:
                        bld_id_csv = str(row.get("BLD_ID", "")).strip()
                        if bld_id_csv:
                            damage_map[bld_id_csv] = row.get("DAMAGE", "").strip()
        except Exception as e_csv:
            print(f"Warning: Could not read damage CSV {building_damage_csv_path}: {e_csv}")
    else:
        print(f"Warning: Building damage CSV not found at {building_damage_csv_path}. DAMAGE field will be empty.")

    # --- 2. Load cube faces metadata ---
    try:
        with open(meta_json_path_obj, 'r') as f_json:
            all_cube_face_records = json.load(f_json)
    except Exception as e_json:
        print(f"Error reading cube faces metadata {meta_json_path_obj}: {e_json}")
        return None

    if not all_cube_face_records:
        print("No cube face records found to sort.")
        return str(sorted_output_root_dir)

    # --- 3. Group records by BLD_ID ---
    records_grouped_by_bld_id = defaultdict(list)
    for record in all_cube_face_records:
        bld_id = str(record.get("BLD_ID", "UnknownBuilding")).strip()
        records_grouped_by_bld_id[bld_id].append(record)
    
    print(f"Sorting {len(all_cube_face_records)} cube face entries into {len(records_grouped_by_bld_id)} building folders...")

    # --- 4. Process each building ---
    for bld_id, building_records in tqdm(records_grouped_by_bld_id.items(), desc="Sorting by Building"):
        safe_bld_id_dirname = bld_id.replace(os.sep, "_").replace(" ", "_").replace(":", "_")
        current_building_output_dir = sorted_output_root_dir / safe_bld_id_dirname
        ensure_dir_exists(current_building_output_dir)

        # Create or update building_info.json
        building_info_json_path = current_building_output_dir / "building_info.json"
        building_info_data = {}
        if building_info_json_path.exists():
            try: # Load existing data to append new views to it
                with open(building_info_json_path, "r") as f:
                    building_info_data = json.load(f)
            except json.JSONDecodeError:
                building_info_data = {}
        
        # Populate static building info if not already present
        if "BLD_ID" not in building_info_data:
            first_record = building_records[0]
            building_info_data = {
                "BLD_ID": first_record.get("BLD_ID"),
                "HEIGHT": first_record.get("HEIGHT"),
                "ELEV": first_record.get("ELEV"),
                "SOURCE": first_record.get("SOURCE"),
                "DATE_": first_record.get("DATE_"),
                "STATUS": first_record.get("STATUS"),
                "building_centroid_lon": first_record.get("building_centroid_lon"),
                "building_centroid_lat": first_record.get("building_centroid_lat"),
                "DAMAGE_assessment": damage_map.get(bld_id, "N/A")
            }

        if "associated_views" not in building_info_data:
            building_info_data["associated_views"] = []

        # Process each view for this building
        for view_record in building_records:
            extracted_faces_dict = view_record.get("extracted_cube_faces", {})
            if not extracted_faces_dict: continue

            view_specific_info = {
                "original_pano_filename": view_record.get("original_pano_filename"),
                "rotated_image_filename": view_record.get("rotated_image_path_relative"),
                "pano_latitude": view_record.get("MAPLatitude"),
                "pano_longitude": view_record.get("MAPLongitude"),
                "view_cube_faces": {},
            }

            for face_name, face_filename in extracted_faces_dict.items():
                # Reconstruct the full path to the source image.
                # It's located relative to the JSON file we are reading.
                source_image_dir = meta_json_path_obj.parent / "cubeface_image_files"
                actual_source_file = source_image_dir / face_filename
                
                if actual_source_file.is_file():
                    destination_face_path = current_building_output_dir / actual_source_file.name
                    try:
                        if move_files:
                            shutil.move(str(actual_source_file), str(destination_face_path))
                        else:
                            shutil.copy2(str(actual_source_file), str(destination_face_path))
                        # Store just the filename in the final JSON, as its location is implied
                        view_specific_info["view_cube_faces"][face_name] = destination_face_path.name
                    except Exception as e_file_op:
                        print(f"Error {'moving' if move_files else 'copying'} file {actual_source_file}: {e_file_op}")
                else:
                    print(f"Warning: Cube face source file not found at {actual_source_file}. Skipping.")
            
            building_info_data["associated_views"].append(view_specific_info)

        # Save the updated building_info.json
        try:
            with open(building_info_json_path, "w") as fp_bld_json:
                json.dump(building_info_data, fp_bld_json, indent=4)
        except Exception as e_bld_json:
            print(f"Error writing building_info.json for BLD_ID {bld_id}: {e_bld_json}")

    print(f"✅ Output sorting complete. Sorted outputs are in → {sorted_output_root_dir}")
    return str(sorted_output_root_dir)
