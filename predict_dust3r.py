#!/usr/bin/env python3
"""
DUSt3R Standalone Prediction Script
-----------------------------------
This script processes multiple images using the DUSt3R model and outputs
all model prediction attributes.
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="DUSt3R Standalone Prediction")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="dust3r_output", help="Output directory for results")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="Image size")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device for inference")
    parser.add_argument("--scene_graph", type=str, default="complete", 
                        choices=["complete", "swin", "oneref"], help="Scene graph type")
    parser.add_argument("--window_size", type=int, default=None, help="Window size for 'swin' scene graph")
    parser.add_argument("--ref_id", type=int, default=0, help="Reference image ID for 'oneref' scene graph")
    parser.add_argument("--iterations", type=int, default=300, help="Number of iterations for global alignment")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="Schedule for global alignment")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="Minimum confidence threshold")
    parser.add_argument("--clean_depth", action="store_true", default=True, help="Clean depth maps")
    parser.add_argument("--mask_sky", action="store_true", default=False, help="Mask sky regions")
    
    return parser.parse_args()

def load_model(weights_path, device='cuda'):
    """Load the DUSt3R model using the AsymmetricCroCo3DStereo class."""
    from dust3r.model import AsymmetricCroCo3DStereo
    
    # Enable TF32 precision on compatible GPUs
    torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
    
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    model.eval()
    return model

def get_all_image_paths(image_dir):
    """Get paths to all images in the directory."""
    extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    return sorted(image_paths)


def configure_scene_graph(scene_graph_type, win_size, ref_id, num_images):
    """Configure the scene graph parameters."""
    if scene_graph_type == "swin":
        if win_size is None:
            win_size = max(1, (num_images - 1) // 2)
        return f"swin-{win_size}"
    elif scene_graph_type == "oneref":
        return f"oneref-{ref_id}"
    else:  # complete
        return "complete"


def save_prediction_attributes(scene, output_dir):
    """Save all prediction attributes to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB images
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    for i, img in enumerate(scene.imgs):
        np.save(os.path.join(rgb_dir, f"image_{i:03d}.npy"), to_numpy(img))
    
    # Save depth maps
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    depthmaps = scene.get_depthmaps()
    for i, depth in enumerate(depthmaps):
        np.save(os.path.join(depth_dir, f"depth_{i:03d}.npy"), to_numpy(depth))
    
    # Save confidence maps
    conf_dir = os.path.join(output_dir, "confidence")
    os.makedirs(conf_dir, exist_ok=True)
    for i, conf in enumerate(scene.im_conf):
        np.save(os.path.join(conf_dir, f"confidence_{i:03d}.npy"), to_numpy(conf))
    
    # Save 3D points
    pts3d_dir = os.path.join(output_dir, "points3d")
    os.makedirs(pts3d_dir, exist_ok=True)
    points3d = scene.get_pts3d()
    for i, pts in enumerate(points3d):
        np.save(os.path.join(pts3d_dir, f"points3d_{i:03d}.npy"), to_numpy(pts))
    
    # Save masks
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    masks = scene.get_masks()
    for i, mask in enumerate(masks):
        np.save(os.path.join(masks_dir, f"mask_{i:03d}.npy"), to_numpy(mask))
    
    # Save camera parameters
    cam_dir = os.path.join(output_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)
    
    # Camera poses (camera-to-world transformations)
    poses = scene.get_im_poses()
    np.save(os.path.join(cam_dir, "camera_poses.npy"), to_numpy(poses))
    
    # Camera intrinsics (focal lengths)
    focals = scene.get_focals()
    np.save(os.path.join(cam_dir, "camera_focals.npy"), to_numpy(focals))
    
    # Save global alignment loss if available
    if hasattr(scene, 'loss_history') and scene.loss_history:
        np.save(os.path.join(output_dir, "global_alignment_loss.npy"), np.array(scene.loss_history))
    
    # Save scene attributes as a dictionary
    scene_attrs = {
        "min_conf_thr": float(scene.min_conf_thr) if hasattr(scene, 'min_conf_thr') else None,
        "num_images": len(scene.imgs),
        "image_size": scene.imgs[0].shape[1:],
    }
    
    np.save(os.path.join(output_dir, "scene_attributes.npy"), scene_attrs)
    
    print(f"All prediction attributes saved to {output_dir}")


def save_scene_as_glb(scene, output_dir, cam_size=0.05, as_pointcloud=False, transparent_cams=False):
    """Save the reconstructed scene as a GLB file."""
    import trimesh
    from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
    from scipy.spatial.transform import Rotation
    from dust3r.utils.device import to_numpy
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, "scene.glb")
    
    # Get scene data
    pts3d = to_numpy(scene.get_pts3d())
    imgs = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals())
    cams2world = to_numpy(scene.get_im_poses())
    mask = to_numpy(scene.get_masks())
    
    # Create trimesh scene
    tri_scene = trimesh.Scene()
    
    # Add point cloud or mesh
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        tri_scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        tri_scene.add_geometry(mesh)
    
    # Add each camera
    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(tri_scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)
    
    # Apply transformation to align with OpenGL coordinate system
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    tri_scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    
    # Export the scene
    print(f"Exporting 3D scene to {glb_path}")
    tri_scene.export(file_obj=glb_path)
    
    return glb_path

def main():
    args = parse_arguments()
    
    # Load the model
    print(f"Loading model on {args.device}...")
    model = load_model(args.weights, args.device)
    
    # Get image paths
    image_paths = get_all_image_paths(args.image_dir)
    if len(image_paths) < 2:
        raise ValueError(f"Found only {len(image_paths)} images. DUSt3R requires at least 2 images.")
    
    print(f"Processing {len(image_paths)} images from {args.image_dir}")
    
    # Load images
    imgs = load_images([str(p) for p in image_paths], size=args.image_size, verbose=True)
    
    # Configure scene graph
    scene_graph = configure_scene_graph(args.scene_graph, args.window_size, args.ref_id, len(imgs))
    print(f"Using scene graph: {scene_graph}")
    
    # Create image pairs
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    
    # Run inference
    print("Running inference...")
    output = inference(pairs, model, args.device, batch_size=1, verbose=True)
    
    # Determine global aligner mode
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    
    # Create and optimize scene
    print("Creating scene...")
    scene = global_aligner(output, device=args.device, mode=mode, verbose=True)
    
    # Run global alignment if necessary
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        print(f"Running global alignment for {args.iterations} iterations...")
        loss = scene.compute_global_alignment(
            init='mst', 
            niter=args.iterations, 
            schedule=args.schedule, 
            lr=0.01
        )
        print(f"Final loss: {loss:.6f}")
    
    # Apply post-processing
    if args.clean_depth:
        print("Cleaning depth maps...")
        scene = scene.clean_pointcloud()
    
    if args.mask_sky:
        print("Masking sky regions...")
        scene = scene.mask_sky()
    
    # Set confidence threshold
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    
    # Save all prediction attributes
    # Save all prediction attributes
    print("Saving prediction attributes...")
    save_prediction_attributes(scene, args.output_dir)
    
    # Save scene as GLB
    print("Exporting 3D scene as GLB...")
    glb_path = save_scene_as_glb(
        scene, 
        args.output_dir, 
        cam_size=0.05, 
        as_pointcloud=False, 
        transparent_cams=False
    )
    print(f"GLB file saved to: {glb_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()