#!/usr/bin/env python3
"""
Phase 1: Scene Reconstruction from Panorama using DUSt3R

This script takes a panoramic image, samples perspective views
(either full sphere or focused on an ROI with controlled overlap),
runs the DUSt3R standalone prediction script on these views,
and loads the resulting camera poses and room geometry.
"""

import os
import subprocess
import json
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from pathlib import Path
import argparse
import sys
import math

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---

# Try importing projection_utils
try:
    from projection_utils import e2p
except ImportError as e:
    print(f"Error importing projection_utils: {e}", file=sys.stderr)
    # ... (rest of error message) ...
    sys.exit(1)

# --- Helper Functions ---

def create_dir_if_not_exists(path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

# get_roi_angular_bbox remains the same
def get_roi_angular_bbox(mask_path: str, pano_width: int, pano_height: int, threshold: int = 128) -> tuple | None:
    """
    Analyzes a mask image to find the angular bounding box of the white ROI.
    (Code identical to previous version)
    """
    print(f"Analyzing ROI mask: {mask_path}")
    try:
        mask_img = Image.open(mask_path)
        if mask_img.size != (pano_width, pano_height):
             print(f"Error: Mask image size {mask_img.size} does not match "
                   f"panorama size ({pano_width}, {pano_height}).", file=sys.stderr)
             return None
        mask_img_gray = ImageOps.grayscale(mask_img)
        mask_np = np.array(mask_img_gray)
        roi_coords = np.argwhere(mask_np > threshold)

        if roi_coords.shape[0] == 0:
            print("Warning: No white pixels found in the mask above threshold.", file=sys.stderr)
            return None

        min_row, min_col = roi_coords.min(axis=0)
        max_row, max_col = roi_coords.max(axis=0)

        # Convert pixel bbox to angular bbox
        min_pitch = 90.0 - (max_row / pano_height) * 180.0
        max_pitch = 90.0 - (min_row / pano_height) * 180.0
        min_yaw = (min_col / pano_width) * 360.0
        max_yaw = (max_col / pano_width) * 360.0

        # Handle potential Yaw wrapping more robustly
        unique_cols = np.unique(roi_coords[:, 1])
        # Check if ROI touches both near-0 and near-360 columns
        col_span = max_col - min_col + 1
        if col_span > pano_width * 0.75: # Heuristic: if pixel span is large...
             # More definitive check: does it contain pixels near both edges?
             near_left_edge = np.any(unique_cols < pano_width * 0.1)
             near_right_edge = np.any(unique_cols > pano_width * 0.9)
             if near_left_edge and near_right_edge:
                 # Find the gap
                 unique_cols_sorted = np.sort(unique_cols)
                 diffs = np.diff(unique_cols_sorted)
                 max_gap_idx = np.argmax(diffs)
                 # If the largest gap is between the last and first element (implies wrap)
                 if unique_cols_sorted[-1] - unique_cols_sorted[0] < max(diffs): # Check standard span vs largest internal gap
                      # The actual max yaw is just before the gap, min yaw is just after
                      max_col_wrapped = unique_cols_sorted[max_gap_idx]
                      min_col_wrapped = unique_cols_sorted[(max_gap_idx + 1) % len(unique_cols_sorted)]
                      min_yaw = (min_col_wrapped / pano_width) * 360.0
                      max_yaw = (max_col_wrapped / pano_width) * 360.0
                      print("  Detected yaw wrap-around (complex case).")


        center_pitch = (min_pitch + max_pitch) / 2.0
        if min_yaw <= max_yaw: center_yaw = (min_yaw + max_yaw) / 2.0
        else: center_yaw = ((min_yaw + max_yaw + 360) / 2.0) % 360.0

        print(f"  ROI angular bbox: yaw=[{min_yaw:.1f}..{max_yaw:.1f}{' (wrap)' if min_yaw > max_yaw else ''}], pitch=[{min_pitch:.1f}..{max_pitch:.1f}]")
        print(f"  ROI center (approx): yaw={center_yaw:.1f}, pitch={center_pitch:.1f}")
        return min_yaw, max_yaw, min_pitch, max_pitch, center_yaw, center_pitch

    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing mask file {mask_path}: {e}", file=sys.stderr)
        return None


# NEW function for overlap-based sampling
def _generate_overlap_roi_view_configs(
    angular_bbox: tuple,
    sample_fov_deg: float,
    overlap_ratio: float
) -> list[dict]:
    """
    Generates view configurations around an ROI with specified overlap.

    Args:
        angular_bbox: Tuple (min_yaw, max_yaw, min_pitch, max_pitch, ...) from get_roi_angular_bbox.
        sample_fov_deg: The field of view for each sampled perspective view.
        overlap_ratio: Desired overlap between adjacent views (0.0 to < 1.0).

    Returns:
        A list of view configuration dictionaries.
    """
    min_yaw, max_yaw, min_pitch, max_pitch, _, _ = angular_bbox

    if not (0 <= overlap_ratio < 1.0):
        raise ValueError("Overlap ratio must be between 0.0 (inclusive) and 1.0 (exclusive).")

    # Calculate the step angle between view centers for the desired overlap
    step_angle = sample_fov_deg * (1.0 - overlap_ratio)
    if step_angle < 1e-3: # Prevent zero or tiny steps if overlap is near 1
        step_angle = 1e-3
        print(f"Warning: Overlap ratio {overlap_ratio} is very high, using minimum step angle {step_angle:.3f} degrees.", file=sys.stderr)
    print(f"  Calculating view grid with FOV={sample_fov_deg} deg, Overlap={overlap_ratio*100:.1f}%, Step Angle={step_angle:.2f} deg")

    # Define the area where view *centers* should be placed.
    # Extend the original ROI bbox by fov/2 so views centered there cover the edge.
    half_fov = sample_fov_deg / 2.0

    # Pitch bounds for view centers
    center_min_pitch = max(-90.0, min_pitch - half_fov)
    center_max_pitch = min( 90.0, max_pitch + half_fov)
    pitch_span = center_max_pitch - center_min_pitch

    # Yaw bounds for view centers (handle wrapping)
    center_min_yaw = (min_yaw - half_fov + 360.0) % 360.0
    center_max_yaw = (max_yaw + half_fov) # Can temporarily exceed 360

    if center_min_yaw <= center_max_yaw % 360.0: # No wrap in the center placement region
        yaw_span = center_max_yaw - center_min_yaw
    else: # Wrap in the center placement region
        yaw_span = (center_max_yaw - center_min_yaw + 360.0)

    # Calculate number of steps needed
    num_pitch_steps = max(1, math.ceil(pitch_span / step_angle)) + 1
    num_yaw_steps = max(1, math.ceil(yaw_span / step_angle)) + 1

    print(f"  Sampling area for view centers: Yaw Span={yaw_span:.1f} deg, Pitch Span={pitch_span:.1f} deg")
    print(f"  Grid size (approx): {num_pitch_steps} (pitch) x {num_yaw_steps} (yaw)")

    # Generate grid points using linspace
    pitch_points = np.linspace(center_min_pitch, center_max_pitch, num_pitch_steps)

    # Generate yaw points carefully handling wrap
    yaw_points_raw = np.linspace(center_min_yaw, center_min_yaw + yaw_span, num_yaw_steps)
    yaw_points = yaw_points_raw % 360.0 # Normalize to [0, 360)

    view_configs = []
    config_idx = 0
    for pitch in pitch_points:
        # Clamp pitch just in case linspace goes slightly out due to float precision
        pitch_clamped = max(-90.0, min(90.0, pitch))
        for yaw in yaw_points:
             name = f"view_roi_{config_idx:03d}_p{int(pitch_clamped):+d}_y{int(yaw):03d}"
             view_configs.append({
                'name': name,
                'fov_deg': sample_fov_deg,
                'yaw_deg': yaw,
                'pitch_deg': pitch_clamped
             })
             config_idx += 1

    print(f"Generated {len(view_configs)} ROI-based view configurations with controlled overlap.")
    return view_configs


# sample_perspective_views remains the same
def sample_perspective_views(
    pano_image_path: str,
    output_dir: str,
    view_configs: list[dict],
    image_size: tuple[int, int],
) -> dict:
    """
    Samples perspective views from a panoramic image based on view configurations.
    (Code identical to previous version)
    """
    print(f"Sampling perspective views from: {pano_image_path}")
    create_dir_if_not_exists(output_dir)
    try:
        pano_img_pil = Image.open(pano_image_path).convert('RGB')
        pano_img_np = np.array(pano_img_pil)
    except FileNotFoundError: raise
    except Exception as e: raise

    sampling_params = {}
    output_height, output_width = image_size
    print(f"Generating {len(view_configs)} perspective views...")
    for config in view_configs:
        name, fov, yaw, pitch = config['name'], config['fov_deg'], config['yaw_deg'], config['pitch_deg']
        print(f"  Generating view: {name} (fov={fov}, yaw={yaw:.1f}, pitch={pitch:.1f})")
        try:
            persp_img_np = e2p(
                e_img=pano_img_np, fov_deg=fov, u_deg=yaw, v_deg=pitch,
                out_hw=(output_height, output_width), roll_deg=0, mode='bilinear'
            )
        except Exception as e:
            print(f"    Error generating perspective view {name}: {e}", file=sys.stderr)
            continue
        persp_img_pil = Image.fromarray(persp_img_np)
        filename = f"{name}.png"
        filepath = os.path.join(output_dir, filename)
        persp_img_pil.save(filepath)
        sampling_params[filename] = {
            'fov_deg': fov, 'yaw_deg': yaw, 'pitch_deg': pitch,
            'original_pano': pano_image_path, 'output_size': image_size
        }
    print(f"Finished sampling {len(sampling_params)} views to {output_dir}")
    return sampling_params

# run_dust3r_predict_script remains the same
def run_dust3r_predict_script(
    predict_script_path: str, image_dir: str, output_dir: str, weights_path: str,
    image_size: int = 512, device: str = 'cuda', scene_graph: str = "complete",
    iterations: int = 300, schedule: str = "linear", min_conf_thr: float = 3.0,
    clean_depth: bool = True, mask_sky: bool = False, extra_args: list = None
) -> bool:
    """Runs the predict_dust3r.py script using subprocess. (Code identical)"""
    # ... (previous implementation) ...
    print("\nRunning DUSt3R prediction script...")
    create_dir_if_not_exists(output_dir)
    if not os.path.exists(predict_script_path): return False # Error printed in calling function now
    if not os.path.exists(weights_path): return False # Error printed in calling function now

    command = [ sys.executable, predict_script_path,
        "--image_dir", image_dir, "--output_dir", output_dir, "--weights", weights_path,
        "--image_size", str(image_size), "--device", device, "--scene_graph", scene_graph,
        "--iterations", str(iterations), "--schedule", schedule, "--min_conf_thr", str(min_conf_thr),
    ]
    if clean_depth: command.append("--clean_depth")
    if mask_sky: command.append("--mask_sky")
    if extra_args: command.extend(extra_args)
    print(f"Executing command:\n{' '.join(command)}")
    try:
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.run( command, check=True, capture_output=False, text=True, env=env)
        print("DUSt3R script finished successfully.")
        return True
    except FileNotFoundError: print(f"Error: Python executable or script not found. Command: {' '.join(command)}", file=sys.stderr); return False
    except subprocess.CalledProcessError as e: print(f"Error: DUSt3R script failed with return code {e.returncode}.", file=sys.stderr); return False
    except Exception as e: print(f"An unexpected error occurred while running DUSt3R script: {e}", file=sys.stderr); return False


# load_dust3r_results remains the same
def load_dust3r_results(results_dir: str, device: torch.device) -> tuple:
    """Loads DUSt3R results. (Code identical)"""
    # ... (previous implementation) ...
    print(f"\nLoading DUSt3R results from: {results_dir}")
    camera_dir = os.path.join(results_dir, "camera")
    cams2world, focals, room_geometry, depth_maps = None, None, None, None
    img_list_path = os.path.join(camera_dir, "image_list.txt")
    ordered_filenames = []
    if os.path.exists(img_list_path):
        with open(img_list_path, 'r') as f: ordered_filenames = [line.strip() for line in f if line.strip()]
        print(f"  Loaded image order from {img_list_path} ({len(ordered_filenames)} images)")
    else: print(f"  Warning: image_list.txt not found.", file=sys.stderr)

    poses_path = os.path.join(camera_dir, "camera_poses.npy")
    if os.path.exists(poses_path):
        cams2world_np = np.load(poses_path)
        cams2world = torch.from_numpy(cams2world_np).float().to(device)
        print(f"  Loaded camera poses: {cams2world.shape}")
        if ordered_filenames and cams2world.shape[0] != len(ordered_filenames): print(f"  Warning: Pose count mismatch.", file=sys.stderr)
    else: print(f"  Warning: Camera poses file not found.", file=sys.stderr)

    focals_path = os.path.join(camera_dir, "camera_focals.npy")
    if os.path.exists(focals_path):
        focals_np = np.load(focals_path)
        focals = torch.from_numpy(focals_np).float().to(device)
        print(f"  Loaded camera focals: {focals.shape}")
        if ordered_filenames and focals.shape[0] != len(ordered_filenames): print(f"  Warning: Focal count mismatch.", file=sys.stderr)
    else: print(f"  Warning: Camera focals file not found.", file=sys.stderr)

    glb_path = os.path.join(results_dir, "scene.glb")
    if os.path.exists(glb_path):
        try:
            room_scene = trimesh.load(glb_path, force='scene')
            if room_scene.geometry:
                geom_key = list(room_scene.geometry.keys())[0]; room_geometry = room_scene.geometry[geom_key]
                if isinstance(room_geometry, (trimesh.Trimesh, trimesh.PointCloud)): room_geometry.vertices = room_geometry.vertices.astype(np.float64)
                print(f"  Loaded room geometry '{geom_key}' ({type(room_geometry)}) with {len(room_geometry.vertices)} vertices/points.")
            else: print(f"  Warning: Loaded scene.glb seems empty.", file=sys.stderr)
        except Exception as e: print(f"  Error loading scene.glb: {e}", file=sys.stderr)
    else: print(f"  Warning: scene.glb file not found.", file=sys.stderr)

    depth_dir = os.path.join(results_dir, "depth"); depth_maps_dict = {}
    if os.path.isdir(depth_dir):
        found_depth_files = list(Path(depth_dir).glob("depth_*.npy"))
        if found_depth_files:
            loaded_count = 0
            for f in found_depth_files:
                try:
                    base_filename_stem = f.stem.replace("depth_", ""); matched_orig_filename = f"{base_filename_stem}.png" # Assume png base name
                    depth_maps_dict[matched_orig_filename] = np.load(f); loaded_count += 1
                except Exception as e: print(f"  Warning: Failed to load depth map {f}: {e}", file=sys.stderr)
            print(f"  Loaded {loaded_count} depth maps from {depth_dir}.")
            if ordered_filenames:
                depth_maps = []; missing_depths = []
                for fname in ordered_filenames:
                    if fname in depth_maps_dict: depth_maps.append(depth_maps_dict[fname])
                    else: depth_maps.append(None); missing_depths.append(fname)
                if missing_depths: print(f"  Warning: Could not find loaded depth maps for: {', '.join(missing_depths)}", file=sys.stderr)
            else: print(f"  Warning: Returning depth maps in arbitrary order.", file=sys.stderr); depth_maps = list(depth_maps_dict.values())
        else: print(f"  No depth map files found in {depth_dir}")
    else: print(f"  Depth map directory not found: {depth_dir}")

    if cams2world is None or focals is None: print("\nWarning: Failed to load essential DUSt3R camera results.", file=sys.stderr)
    return cams2world, focals, room_geometry, depth_maps, ordered_filenames

# --- Main Phase 1 Function ---

def run_phase1(
    pano_image_path: str,
    output_base_dir: str,
    dust3r_weights_path: str,
    predict_script_path: str,
    roi_mask_path: str | None = None,
    view_overlap: float = 0.5, # <-- New: Overlap ratio for ROI
    # num_roi_views: int = 16, # <-- Removed: No longer target specific number for ROI
    fov_deg: float = 90.0,
    num_yaw_steps: int = 8,
    num_pitch_levels: int = 3,
    pitch_angles_deg: list | None = None,
    dust3r_image_size: int = 512,
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> dict | None:
    """
    Executes Phase 1: Sample views, run DUSt3R, load results.
    Supports full sphere or ROI-focused sampling with controlled overlap.

    Args:
        pano_image_path: Path to the input panoramic image.
        output_base_dir: Base directory for phase 1 outputs.
        dust3r_weights_path: Path to the DUSt3R model weights (.ckpt).
        predict_script_path: Path to the `predict_dust3r.py` script.
        roi_mask_path: Optional path to ROI mask image. Enables ROI sampling.
        view_overlap: Desired overlap ratio (0 to <1) for ROI sampling. Ignored otherwise.
        fov_deg: Field of View for sampling perspective views.
        num_yaw_steps: Number of views horizontally (full sphere sampling).
        num_pitch_levels: Number of pitch levels (full sphere sampling).
        pitch_angles_deg: Explicit pitch angles (full sphere sampling).
        dust3r_image_size: Image size for DUSt3R processing.
        device_str: Device string ('cuda' or 'cpu').

    Returns:
        A dictionary containing results, or None on critical failure.
    """
    print("-" * 30); print("Executing Phase 1: Scene Reconstruction"); print("-" * 30)

    # --- Pre-checks ---
    if not os.path.isfile(pano_image_path): print(f"Error: Pano image not found: {pano_image_path}", file=sys.stderr); return None
    if not os.path.isfile(dust3r_weights_path): print(f"Error: DUSt3R weights not found: {dust3r_weights_path}", file=sys.stderr); return None
    if not os.path.isfile(predict_script_path): print(f"Error: predict_dust3r.py script not found: {predict_script_path}", file=sys.stderr); return None
    if roi_mask_path and not os.path.isfile(roi_mask_path): print(f"Error: ROI mask file not found: {roi_mask_path}", file=sys.stderr); return None

    # --- Define Directories ---
    sampled_views_dir = os.path.join(output_base_dir, "sampled_views")
    dust3r_results_dir = os.path.join(output_base_dir, "dust3r_results")
    sampling_params_file = os.path.join(output_base_dir, "sampling_params.json")

    # --- Get Panorama Dimensions ---
    try:
        with Image.open(pano_image_path) as img: pano_width, pano_height = img.size
    except Exception as e: print(f"Error reading panorama dimensions: {e}", file=sys.stderr); return None

    # --- Generate View Sampling Configurations ---
    view_configs = []
    sampling_mode = "Full Sphere"

    if roi_mask_path:
        sampling_mode = "ROI Focused (Overlap Control)"
        angular_bbox = get_roi_angular_bbox(roi_mask_path, pano_width, pano_height)
        if angular_bbox:
            try:
                view_configs = _generate_overlap_roi_view_configs(
                    angular_bbox=angular_bbox,
                    sample_fov_deg=fov_deg,
                    overlap_ratio=view_overlap # Pass the overlap ratio
                )
            except ValueError as e:
                 print(f"Error generating ROI views: {e}", file=sys.stderr)
                 return None
        else:
            print("Error: Could not determine ROI from mask. Aborting.", file=sys.stderr)
            return None
    else:
        # Fallback to Full Sphere Sampling (original logic)
        # ... (Full sphere generation logic remains exactly the same as previous version) ...
        yaws_deg = np.linspace(0, 360, num_yaw_steps, endpoint=False)
        if pitch_angles_deg is not None: pitches_deg = sorted(list(set(pitch_angles_deg)))
        else:
            if num_pitch_levels < 1: print("Error: num_pitch_levels must be >= 1.", file=sys.stderr); return None
            if num_pitch_levels == 1: pitches_deg = [0.0]
            else: pitches_deg = np.linspace(-90.0, 90.0, num_pitch_levels)
        processed_poles = {'+90': False, '-90': False}; config_idx = 0
        for pitch in pitches_deg:
            is_pole = abs(abs(pitch) - 90.0) < 1e-3
            if is_pole:
                pole_key = '+90' if pitch > 0 else '-90'
                if not processed_poles[pole_key]:
                    yaw = 0.0; name = f"view_p{int(pitch):+d}_y{int(yaw):03d}"
                    view_configs.append({'name': name, 'fov_deg': fov_deg, 'yaw_deg': yaw, 'pitch_deg': pitch})
                    processed_poles[pole_key] = True; config_idx += 1
                continue
            for yaw in yaws_deg:
                 name = f"view_p{int(pitch):+d}_y{int(yaw):03d}"
                 view_configs.append({'name': name, 'fov_deg': fov_deg, 'yaw_deg': yaw, 'pitch_deg': pitch})
                 config_idx += 1


    print(f"Using {sampling_mode} sampling strategy.")
    if not view_configs: print("Error: No view configurations were generated.", file=sys.stderr); return None
    print(f"Targeting {len(view_configs)} view configurations.")

    # --- 1. Sample Perspective Views ---
    try:
        actual_sampling_params = sample_perspective_views(
            pano_image_path, sampled_views_dir, view_configs,
            image_size=(dust3r_image_size, dust3r_image_size)
        )
        with open(sampling_params_file, 'w') as f: json.dump(actual_sampling_params, f, indent=4)
        print(f"Saved actual sampling parameters for {len(actual_sampling_params)} views to {sampling_params_file}")
    except Exception as e: print(f"Error during perspective view sampling: {e}", file=sys.stderr); return None

    if not actual_sampling_params: print("Error: No perspective views were successfully generated.", file=sys.stderr); return None

    # --- 2. Run DUSt3R Prediction Script ---
    success = run_dust3r_predict_script(
        predict_script_path=predict_script_path, image_dir=sampled_views_dir,
        output_dir=dust3r_results_dir, weights_path=dust3r_weights_path,
        image_size=dust3r_image_size, device=device_str,
        scene_graph="complete", iterations=300, clean_depth=True,
    )
    if not success: print("Error: DUSt3R prediction script execution failed.", file=sys.stderr); return None

    # --- 3. Load DUSt3R Results ---
    pytorch_device = torch.device(device_str)
    try:
        cams2world, focals, room_geometry, depth_maps, image_filenames = load_dust3r_results(
            dust3r_results_dir, pytorch_device
        )
        with open(sampling_params_file, 'r') as f: loaded_sampling_params = json.load(f)
    except Exception as e: print(f"Error loading DUSt3R results: {e}", file=sys.stderr); return None

    # --- Validation ---
    if cams2world is None or focals is None: print("Error: Failed to load essential camera results.", file=sys.stderr); return None
    if room_geometry is None: print("Warning: Failed to load room geometry.", file=sys.stderr)
    # ... (Validation checks using image_filenames or fallback remain the same) ...
    if image_filenames:
        num_images = len(image_filenames)
        if cams2world.shape[0]!=num_images: print(f"Warn: Pose({cams2world.shape[0]})!=ImgList({num_images})", file=sys.stderr)
        if focals.shape[0]!=num_images: print(f"Warn: Focal({focals.shape[0]})!=ImgList({num_images})", file=sys.stderr)
        if depth_maps and len(depth_maps)!=num_images: print(f"Warn: Depth({len(depth_maps)})!=ImgList({num_images})", file=sys.stderr)
        if not all(f in loaded_sampling_params for f in image_filenames): print(f"Warn: ImgList/SamplingParams mismatch", file=sys.stderr)
    else: # Fallback check
        num_sampled = len(loaded_sampling_params)
        if cams2world.shape[0]!=num_sampled: print(f"Warn: Pose({cams2world.shape[0]})!=Sampled({num_sampled})", file=sys.stderr)


    print("-" * 30); print("Phase 1 Completed Successfully."); print("-" * 30)
    return {
        'cams2world': cams2world, 'focals': focals, 'room_geometry': room_geometry,
        'sampling_params': loaded_sampling_params, 'image_filenames': image_filenames,
        'depth_maps': depth_maps, 'dust3r_results_dir': dust3r_results_dir,
        'sampled_views_dir': sampled_views_dir
    }


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1: DUSt3R Scene Reconstruction from Panorama")
    parser.add_argument("--pano_image", type=str, required=True, help="Path to the input panoramic image (.jpg, .png)")
    parser.add_argument("--output_dir", type=str, default="phase1_output", help="Base directory for Phase 1 outputs")
    parser.add_argument("--dust3r_weights", type=str, required=True, help="Path to DUSt3R model weights (.ckpt)")
    parser.add_argument("--predict_script", type=str, default="predict_dust3r.py", help="Path to the predict_dust3r.py script")

    # Sampling parameters - Mode selection
    parser.add_argument("--roi_mask", type=str, default=None, help="Path to ROI mask image (same size as pano). Enables ROI-focused sampling.")
    parser.add_argument("--view_overlap", type=float, default=0.5, help="Desired view overlap ratio [0.0, 1.0) for ROI sampling. E.g., 0.5 for 50%% overlap.")

    # Sampling parameters - Common
    parser.add_argument("--fov", type=float, default=90.0, help="Field of View (degrees) for sampling perspective views")

    # Sampling parameters - Full Sphere (used if --roi_mask is NOT provided)
    parser.add_argument("--num_yaw_steps", type=int, default=8, help="Number of views horizontally (full sphere sampling)")
    parser.add_argument("--num_pitch_levels", type=int, default=3, help="Number of pitch levels vertically (full sphere sampling)")
    parser.add_argument("--pitch_angles", type=float, nargs='+', default=None, help="Explicit pitch angles (degrees) for full sphere sampling.")

    # DUSt3R parameters
    parser.add_argument("--img_size", type=int, default=512, help="Image size for DUSt3R processing")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.roi_mask and not (0 <= args.view_overlap < 1.0):
        parser.error("--view_overlap must be between 0.0 (inclusive) and 1.0 (exclusive) when using --roi_mask.")
    # ... (other validation checks remain the same) ...
    if not args.roi_mask:
        if args.pitch_angles is None:
            if args.num_pitch_levels < 1: parser.error("--num_pitch_levels must be >= 1 for full sphere sampling.")
        else:
            if any(p < -90 or p > 90 for p in args.pitch_angles): parser.error("--pitch_angles must be between -90 and 90 degrees.")

    # --- Run Phase 1 ---
    phase1_results = run_phase1(
        pano_image_path=args.pano_image,
        output_base_dir=args.output_dir,
        dust3r_weights_path=args.dust3r_weights,
        predict_script_path=args.predict_script,
        roi_mask_path=args.roi_mask,
        view_overlap=args.view_overlap, # Pass overlap ratio
        fov_deg=args.fov,
        num_yaw_steps=args.num_yaw_steps,
        num_pitch_levels=args.num_pitch_levels,
        pitch_angles_deg=args.pitch_angles,
        dust3r_image_size=args.img_size,
        device_str=args.device
    )

    # --- Report Results ---
    if phase1_results:
        print("\n--- Phase 1 Results Summary ---")
        # ... (Result reporting section remains the same) ...
        num_processed_views = len(phase1_results.get('image_filenames', []))
        print(f"Number of views processed by DUSt3R: {num_processed_views}")
        if phase1_results['cams2world'] is not None: print(f"Camera Poses (cams2world) shape: {phase1_results['cams2world'].shape}")
        if phase1_results['focals'] is not None: print(f"Focal Lengths (focals) shape: {phase1_results['focals'].shape}")
        if phase1_results['room_geometry']:
            geom_type = type(phase1_results['room_geometry']).__name__
            num_verts = len(phase1_results['room_geometry'].vertices)
            print(f"Room Geometry: Loaded {geom_type} with {num_verts} vertices/points")
        else: print("Room Geometry: Not loaded (or loading failed).")
        print(f"Number of sampling parameters recorded: {len(phase1_results['sampling_params'])}")
        if phase1_results['depth_maps']:
             valid_depths = sum(1 for d in phase1_results['depth_maps'] if d is not None)
             print(f"Number of depth maps loaded (matching order): {valid_depths} / {num_processed_views}")
        print(f"DUSt3R results saved in: {phase1_results['dust3r_results_dir']}")
        print(f"Sampled views saved in: {phase1_results['sampled_views_dir']}")
        if phase1_results.get('image_filenames'): print(f"Order of processed images (from image_list.txt): {phase1_results['image_filenames'][:5]}...")

    else:
        print("\nPhase 1 execution failed.")
        sys.exit(1)