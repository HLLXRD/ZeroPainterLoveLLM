import os
import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    # Use FoVPerspectiveCameras for easier 90-degree FOV setting
    FoVPerspectiveCameras,
    PerspectiveCameras, # Keep for potential alternative
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams
)
# No need for these transforms if only optimizing position
# from pytorch3d.transforms import RotateAxisAngle, Transform3d

# Use projection_utils for e2c
try:
    from projection_utils import e2c # Import the main function needed
except ImportError as e:
    print(f"Error importing projection_utils: {e}")
    print("Please ensure projection_utils.py is in the same directory or Python path.")
    exit()

# --- Configuration ---
OBJ_PATH = "/root/normalized_model.obj"  # REQUIRED: Update this path
MASK_PATH = "/root/silhouette.png" # REQUIRED: Update this path
OUTPUT_DIR = "optimization_output_single_pos" # Changed output dir name
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cubemap settings
FACE_W = 256  # Resolution of each cubemap face
CUBE_FORMAT_TARGET = "list" # Use 'list' for easier handling

# Camera settings
# Initial guess for the single camera position
INITIAL_CAMERA_POS = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE) # Start at origin relative to normalized mesh
# Alt: If you have a better guess:
# INITIAL_CAMERA_POS = torch.tensor([[0.1, 0.2, -0.05]], device=DEVICE)

# Renderer settings
IMAGE_SIZE = FACE_W # Should match cubemap face width
# SIGMA = 1e-5 # Sigma for SoftSilhouetteShader, smaller values -> sharper edges
# GAMMA = 1e-5 # Gamma for SoftSilhouetteShader
SIGMA = 1e-4 # Slightly larger sigma might help gradients initially
GAMMA = 1e-4
FACES_PER_PIXEL = 5 # Increased slightly for potentially better anti-aliasing with soft shader
BLUR_RADIUS = 0.0 # SoftSilhouetteShader handles the edge softness

# Optimization settings
LEARNING_RATE = 0.01 # Might need smaller LR for position optimization
NUM_ITERATIONS = 2000 # More iterations might be needed
LOG_INTERVAL = 20

# --- Helper Functions ---

def load_mesh(obj_path):
    """Loads mesh, centers it, and scales it to fit in unit cube."""
    verts, faces_idx, _ = load_obj(obj_path, device=DEVICE)
    faces = faces_idx.verts_idx

    # Center and scale mesh
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    # Handle cases where scale might be zero (e.g., a single point)
    if scale > 1e-8:
        verts = verts / scale
    else:
        print("Warning: Mesh scale is near zero.")
        scale = 1.0 # Avoid division by zero

    print(f"Mesh loaded: {verts.shape[0]} vertices, {faces.shape[0]} faces")
    print(f"Mesh centered at origin, scaled to max extent: {scale:.3f}")

    # White textures
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None] * 0.9) # BxVxC

    return Meshes(verts=[verts], faces=[faces], textures=textures)

def load_and_prepare_target_mask(mask_path, face_w, cube_format, device):
    """Loads equirectangular mask, converts to cubemap, and prepares target tensor."""
    try:
        print(f"Loading mask: {mask_path}")
        mask_img_pil = Image.open(mask_path).convert("L") # Load as grayscale
        mask_e_np = np.array(mask_img_pil) / 255.0 # Normalize to [0, 1]
        print(f"Equirectangular mask loaded, shape: {mask_e_np.shape}")

        # Use the provided e2c function
        print(f"Converting equirectangular mask to cubemap (format: {cube_format}, face_w: {face_w})...")
        cubemap_target_np = e2c(
            mask_e_np,
            face_w=face_w,
            mode="bilinear",
            cube_format=cube_format
        )
        print("Conversion complete.")

        if cube_format == "list":
            # Convert list of numpy arrays to list of PyTorch tensors
            cubemap_target = [torch.tensor(face, dtype=torch.float32, device=device) for face in cubemap_target_np]
            print(f"Target cubemap created as list of {len(cubemap_target)} tensors, each shape: {cubemap_target[0].shape}")
            # Add channel dimension if missing (e.g., for BCE loss) and ensure correct dims [H, W, C]
            processed_target = []
            for face_tensor in cubemap_target:
                if face_tensor.ndim == 2: # H, W -> H, W, 1
                    face_tensor = face_tensor.unsqueeze(-1)
                elif face_tensor.ndim == 3 and face_tensor.shape[-1] != 1: # H, W, C!=1 -> H, W, 1 (take first channel or average if needed)
                     print(f"Warning: Target face has unexpected channels {face_tensor.shape[-1]}. Taking first channel.")
                     face_tensor = face_tensor[..., 0:1]
                processed_target.append(face_tensor)
            cubemap_target = processed_target
            print(f"Target cubemap shapes after channel check: {[f.shape for f in cubemap_target]}")
            # Should be [(H, W, 1), (H, W, 1), ...]

        # Add other formats ('horizon', 'dice', 'dict') if needed
        else:
            raise NotImplementedError(f"Cube format '{cube_format}' handling not fully implemented for target.")

        return cubemap_target

    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        exit()
    except Exception as e:
        print(f"Error processing mask: {e}")
        # raise e # Re-raise to see the full traceback if needed
        exit()

def create_single_pos_cubemap_cameras(position, fov, image_size, device):
    """
    Creates 6 cameras at the *same* given position, oriented for cubemap faces
    (F, R, B, L, U, D) from that single point.
    Uses FoVPerspectiveCameras for straightforward 90-degree FoV.

    Args:
        position (torch.Tensor): Shape (1, 3) tensor for the camera location.
        fov (float): Field of view in degrees (should be 90 for standard cubemap faces).
        image_size (int): The width/height of the square camera image.
        device: The torch device.

    Returns:
        list[FoVPerspectiveCameras]: A list of 6 cameras, one for each cubemap face.
    """
    cameras = []
    # Ensure position is on the correct device and has batch dim [1, 3]
    eye = position.to(device).view(1, 3)

    # Define look-at points and up vectors relative to the 'eye' position
    # These define the *direction* the camera is looking.
    # Using PyTorch3D conventions: +Y Up, +X Right, -Z Forward (view direction)
    # Camera points *towards* the 'at' point.
    # F: Look towards -Z world -> at = eye + [0, 0, -1]
    # R: Look towards +X world -> at = eye + [1, 0, 0]
    # B: Look towards +Z world -> at = eye + [0, 0, 1]
    # L: Look towards -X world -> at = eye + [-1, 0, 0]
    # U: Look towards +Y world -> at = eye + [0, 1, 0]
    # D: Look towards -Y world -> at = eye + [0, -1, 0]

    # Up vectors need care, especially for Up/Down views.
    # Standard convention:
    # F, R, B, L use +Y as up.
    # U uses -Z as up (looking up, top of camera points "backwards").
    # D uses +Z as up (looking down, top of camera points "forwards").

    directions = [
        # (look_direction, up_vector)
        (torch.tensor([[0.0, 0.0, -1.0]], device=device), torch.tensor([[0.0, 1.0, 0.0]], device=device)), # Front (-Z)
        (torch.tensor([[1.0, 0.0, 0.0]], device=device), torch.tensor([[0.0, 1.0, 0.0]], device=device)),  # Right (+X)
        (torch.tensor([[0.0, 0.0, 1.0]], device=device), torch.tensor([[0.0, 1.0, 0.0]], device=device)),   # Back (+Z)
        (torch.tensor([[-1.0, 0.0, 0.0]], device=device), torch.tensor([[0.0, 1.0, 0.0]], device=device)), # Left (-X)
        (torch.tensor([[0.0, 1.0, 0.0]], device=device), torch.tensor([[0.0, 0.0, -1.0]], device=device)), # Up (+Y), Up is -Z
        (torch.tensor([[0.0, -1.0, 0.0]], device=device), torch.tensor([[0.0, 0.0, 1.0]], device=device)),  # Down (-Y), Up is +Z
    ]

    face_order = ["F", "R", "B", "L", "U", "D"] # Corresponds to e2c list order

    for i, (direction, up) in enumerate(directions):
        # Calculate the 'at' point
        at = eye + direction

        # Use look_at_view_transform to get R and T
        # R, T transform world points to view coordinates.
        # T is the camera *position* in world coords, but negated and rotated into view space.
        # What we need for the Camera class is R (world-to-view rotation) and T_cam (camera world position).
        # look_at_view_transform gives us exactly this R and the correct T for the camera constructor.
        R, T_cam = look_at_view_transform(eye=eye, at=at, up=up, device=device)
        # Note: T_cam from look_at_view_transform is the camera *translation* component needed by PyTorch3D cameras.
        # It's not just the 'eye' position directly, but the result of the look_at calculation.

        # Create the camera for this face
        cameras.append(FoVPerspectiveCameras(
            fov=fov,
            znear=0.01, # Adjust near/far clipping planes if needed
            zfar=100.0,
            R=R,
            T=T_cam,
            device=device,
            # image_size must be set if aspect_ratio isn't 1 or if needed downstream
            # It doesn't directly affect camera parameters here but useful context
            aspect_ratio=1.0 # Since image_size W=H
        ))
        # print(f"Face {face_order[i]}: Eye={eye.cpu().numpy()}, At={at.cpu().numpy()}, Up={up.cpu().numpy()}")
        # print(f"  R={R.cpu().numpy().round(2)}")
        # print(f"  T_cam={T_cam.cpu().numpy().round(2)}")


    return cameras


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(OBJ_PATH):
        print(f"Error: OBJ file not found at {OBJ_PATH}")
        exit()
    if not os.path.exists(MASK_PATH):
        print(f"Error: Mask file not found at {MASK_PATH}")
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Mesh
    mesh = load_mesh(OBJ_PATH) # Mesh centered at origin, scaled to unit cube

    # 2. Load and Prepare Target Mask
    target_cubemap_list = load_and_prepare_target_mask(
        MASK_PATH, FACE_W, CUBE_FORMAT_TARGET, DEVICE
    )
    # Target should be a list of 6 [H, W, 1] tensors on DEVICE

    # 3. Initialize Camera Position (Single Point, Optimizable)
    # Detach initial position if it came from requires_grad=True tensor
    camera_position = torch.nn.Parameter(INITIAL_CAMERA_POS.clone().detach().to(DEVICE))


    # 4. Setup Renderer
    raster_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=BLUR_RADIUS, # Use 0.0 if SoftSilhouetteShader handles all blending
        faces_per_pixel=FACES_PER_PIXEL,
        perspective_correct=True,
        cull_backfaces=False, # Usually False for 360 views (see inside/outside)
                              # Set True if mesh is closed and view is external
    )

    blend_params = BlendParams(sigma=SIGMA, gamma=GAMMA)
    silhouette_shader = SoftSilhouetteShader(blend_params=blend_params)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=silhouette_shader
    )

    # 5. Optimization Loop
    optimizer = torch.optim.Adam([camera_position], lr=LEARNING_RATE)
    # Loss function (Binary Cross Entropy or MSE can work for silhouettes)
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss() # MSE is sometimes more stable

    print("\nStarting optimization...")
    for i in range(NUM_ITERATIONS):
        optimizer.zero_grad()

        # Create the 6 cameras VIEWING FROM the current position estimate
        # All 6 cameras will share the optimized 'camera_position' as their origin (eye point)
        cubemap_cameras = create_single_pos_cubemap_cameras(
            position=camera_position,
            fov=90.0, # Standard cubemap FOV
            image_size=IMAGE_SIZE,
            device=DEVICE
        )

        rendered_faces = []
        total_loss = 0.0

        # Render each face
        for face_idx in range(6):
            current_camera = cubemap_cameras[face_idx]
            target_face = target_cubemap_list[face_idx] # Shape [H, W, 1]

            # Render silhouette: result is [batch, H, W, 4] RGBA
            # Need image_size context if not provided to camera directly
            rendered_output = renderer(mesh.extend(1), cameras=current_camera, image_size=IMAGE_SIZE) # Pass image_size here too if needed
            rendered_alpha = rendered_output[..., 3] # Get alpha channel, shape [1, H, W]

            # Ensure target and prediction have same shape for loss: [H, W, 1] vs [1, H, W] -> [H, W, 1]
            rendered_alpha_permuted = rendered_alpha.squeeze(0).unsqueeze(-1) # -> [H, W, 1]

            # Calculate loss for this face
            # Ensure target_face is also [H, W, 1]
            loss_face = loss_fn(rendered_alpha_permuted, target_face)
            total_loss += loss_face
            rendered_faces.append(rendered_alpha_permuted) # Store [H, W, 1] tensor

        # Average loss over faces
        loss = total_loss / 6.0

        # Backpropagate and optimize
        loss.backward()

        # Optional: Gradient clipping can help stabilize optimization
        # torch.nn.utils.clip_grad_norm_(camera_position, max_norm=1.0)

        optimizer.step()

        # Log progress
        if i % LOG_INTERVAL == 0 or i == NUM_ITERATIONS - 1:
            print(f"Iteration {i}/{NUM_ITERATIONS}, Loss: {loss.item():.6f}")
            print(f"  Camera Position: {camera_position.data.cpu().numpy().round(4)}")

            # Save intermediate rendered cubemap vs target
            if i % (LOG_INTERVAL * 5) == 0 or i == NUM_ITERATIONS - 1 : # Save less frequently
                try:
                    from torchvision.utils import save_image
                    # Permute to [C, H, W] for save_image
                    rendered_list_chw = [f.permute(2, 0, 1) for f in rendered_faces]
                    target_list_chw = [f.permute(2, 0, 1) for f in target_cubemap_list]

                    rendered_horizon = torch.cat(rendered_list_chw, dim=2) # C, H, W*6
                    target_horizon = torch.cat(target_list_chw, dim=2)

                    # Ensure tensors are on CPU for saving if needed by save_image
                    comparison = torch.cat([target_horizon.cpu(), rendered_horizon.cpu()], dim=1) # Stack target above render
                    save_image(comparison, os.path.join(OUTPUT_DIR, f"render_vs_target_{i:04d}.png"))
                except ImportError:
                    if i == 0: print("Install torchvision to save intermediate images.")
                except Exception as e_save:
                    print(f"Could not save image: {e_save}")


    print("\nOptimization finished.")
    final_position = camera_position.data.cpu().numpy()
    print(f"Final Optimized Camera Position: {final_position.flatten()}")

    # To use the result:
    # final_cameras = create_single_pos_cubemap_cameras(camera_position.data, 90.0, IMAGE_SIZE, DEVICE)
    # Now you can render the final cubemap using 'final_cameras' and the 'renderer'.