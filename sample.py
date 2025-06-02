
#!/usr/bin/env python3
"""
sample.py – Samples video frames based on distance using mapillary_tools.
"""
import os
import subprocess
import sys
import json
from utils import ensure_dir_exists

def sample_video_by_distance(
    video_path: str,
    base_output_dir: str,
    distance_m: float,
    mapillary_tools_path: str = "mapillary_tools"
):
    """
    Uses mapillary_tools to sample video frames every `distance_m` meters.

    Args:
        video_path: Path to the input video file (e.g., .mp4).
        base_output_dir: Base directory where all pipeline outputs will be stored.
                         Sampled images will be in a subdirectory.
        distance_m: Sampling interval in meters.
        mapillary_tools_path: Path to the mapillary_tools executable.

    Returns:
        A tuple (sampled_images_dir, mapillary_image_description_json_path)
        or (None, None) if an error occurs.
    """
    sampled_images_dir = os.path.join(base_output_dir, "01_sampled_images")
    ensure_dir_exists(sampled_images_dir)
    
    # The mapillary_tools process command writes the JSON to the image directory
    mapillary_image_description_json_path = os.path.join(sampled_images_dir, "mapillary_image_description.json")

    # 1) Run the Mapillary video sampler
    cmd_sample = [
        mapillary_tools_path,
        "video_process",
        video_path,
        sampled_images_dir, # mapillary_tools puts images directly here
        f"--video_sample_distance={distance_m}"
    ]
    print(f"Running video sampling: {' '.join(cmd_sample)}")
    try:
        subprocess.run(cmd_sample, check=True, capture_output=True, text=True)
        print(f"Video sampling successful. Images in: {sampled_images_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during mapillary_tools video_process: {e}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
        print(f"Error: mapillary_tools executable not found at '{mapillary_tools_path}'. Please check the path.", file=sys.stderr)
        return None, None

    # 2) Geotag / process the sampled images (this also creates mapillary_image_description.json)
    # This step is often crucial for preparing images for further Mapillary processing or for extracting metadata.
    # If mapillary_tools video_process already created a suitable mapillary_image_description.json,
    # this might be redundant or might refine it. Check mapillary_tools documentation for your version.
    cmd_process = [
        mapillary_tools_path,
        "process",
        sampled_images_dir
    ]
    print(f"Running image processing: {' '.join(cmd_process)}")
    try:
        subprocess.run(cmd_process, check=True, capture_output=True, text=True)
        print(f"Image processing successful. Metadata JSON: {mapillary_image_description_json_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during mapillary_tools process: {e}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        # If the JSON file was created by video_process, we might still be able to proceed
        if not os.path.exists(mapillary_image_description_json_path):
            return None, None
        print("Warning: 'mapillary_tools process' failed, but 'mapillary_image_description.json' might exist from 'video_process'.", file=sys.stderr)


    if not os.path.exists(mapillary_image_description_json_path):
        print(f"Error: 'mapillary_image_description.json' not found in {sampled_images_dir} after processing.", file=sys.stderr)
        return None, None
        
    print(f"Done! Sampled images are in {sampled_images_dir}")
    print(f"Image description JSON at {mapillary_image_description_json_path}")
    return sampled_images_dir, mapillary_image_description_json_path