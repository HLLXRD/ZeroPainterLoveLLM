# Combined utilities for equirectangular/cubemap conversion and related geometry
# Based on the provided utils.py, equirectangulartocubemap.py, and e2p.py snippets.

from collections.abc import Sequence
from enum import IntEnum
from functools import lru_cache
from numbers import Real # Added for e2p
from typing import Any, Literal, Optional, TypeVar, Union, overload, Dict, List, Tuple # Added Tuple for e2p

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

try:
    import cv2  # pyright: ignore[reportMissingImports]
except ImportError:
    cv2 = None

# --- Constants and Type Definitions ---

_mode_to_order = {
    "nearest": 0,
    "linear": 1,
    "bilinear": 1,
    "biquadratic": 2,
    "quadratic": 2,
    "quad": 2,
    "bicubic": 3,
    "cubic": 3,
    "biquartic": 4,
    "quartic": 4,
    "biquintic": 5,
    "quintic": 5,
}

CubeFormat = Literal["horizon", "list", "dict", "dice"]
InterpolationMode = Literal[
    "nearest",
    "linear",
    "bilinear",
    "biquadratic",
    "quadratic",
    "quad",
    "bicubic",
    "cubic",
    "biquartic",
    "quartic",
    "biquintic",
    "quintic",
]
DType = TypeVar("DType", bound=np.generic, covariant=True)
_CACHE_SIZE = 8


class Face(IntEnum):
    """Face type indexing for numpy vectorization."""

    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
    UP = 4
    DOWN = 5


class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2

# --- Core Utility Functions (from original utils.py) ---

def mode_to_order(mode: InterpolationMode) -> int:
    """Convert a human-friendly interpolation string to integer equivalent.

    Parameters
    ----------
    mode: str
        Human-friendly interpolation string.

    Returns
    -------
    The order of the spline interpolation
    """
    try:
        return _mode_to_order[mode.lower()]
    except KeyError:
        raise ValueError(f'Unknown mode "{mode}".') from None


def slice_chunk(index: int, width: int, offset=0):
    start = index * width + offset
    return slice(start, start + width)


# @lru_cache(_CACHE_SIZE) # Cache disabled for mutable numpy arrays if not careful
def xyzcube(face_w: int) -> NDArray[np.float32]:
    """
    Return the xyz coordinates of the unit cube in [F R B L U D] format.

    Parameters
    ----------
    face_w: int
        Specify the length of each face of the cubemap.

    Returns
    -------
    out: ndarray
        An array object with dimension (face_w, face_w * 6, 3)
        which store the each face of numalized cube coordinates.
        The cube is centered at the origin so that each face k
        in out has range [-0.5, 0.5] x [-0.5, 0.5].
    """
    out = np.empty((face_w, face_w * 6, 3), np.float32)

    # Create coordinates once and reuse
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    x, y = np.meshgrid(rng, -rng)

    # Pre-compute flips
    x_flip = np.flip(x, 1)
    y_flip = np.flip(y, 0)

    def face_slice(index):
        return slice_chunk(index, face_w)

    # Front face (z = 0.5)
    out[:, face_slice(Face.FRONT), Dim.X] = x
    out[:, face_slice(Face.FRONT), Dim.Y] = y
    out[:, face_slice(Face.FRONT), Dim.Z] = 0.5

    # Right face (x = 0.5)
    out[:, face_slice(Face.RIGHT), Dim.X] = 0.5
    out[:, face_slice(Face.RIGHT), Dim.Y] = y
    out[:, face_slice(Face.RIGHT), Dim.Z] = x_flip

    # Back face (z = -0.5)
    out[:, face_slice(Face.BACK), Dim.X] = x_flip
    out[:, face_slice(Face.BACK), Dim.Y] = y
    out[:, face_slice(Face.BACK), Dim.Z] = -0.5

    # Left face (x = -0.5)
    out[:, face_slice(Face.LEFT), Dim.X] = -0.5
    out[:, face_slice(Face.LEFT), Dim.Y] = y
    out[:, face_slice(Face.LEFT), Dim.Z] = x

    # Up face (y = 0.5)
    out[:, face_slice(Face.UP), Dim.X] = x
    out[:, face_slice(Face.UP), Dim.Y] = 0.5
    out[:, face_slice(Face.UP), Dim.Z] = y_flip

    # Down face (y = -0.5)
    out[:, face_slice(Face.DOWN), Dim.X] = x
    out[:, face_slice(Face.DOWN), Dim.Y] = -0.5
    out[:, face_slice(Face.DOWN), Dim.Z] = y

    # Since we are using lru_cache, we want the return value to be immutable.
    # out.setflags(write=False) # Disabled for now
    return out


# @lru_cache(_CACHE_SIZE) # Cache disabled for mutable numpy arrays if not careful
def equirect_uvgrid(h: int, w: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi / 2, -np.pi / 2, num=h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    # Since we are using lru_cache, we want the return value to be immutable.
    # uu.setflags(write=False) # Disabled for now
    # vv.setflags(write=False) # Disabled for now
    return uu, vv  # pyright: ignore[reportReturnType]


# @lru_cache(_CACHE_SIZE) # Cache disabled for mutable numpy arrays if not careful
def equirect_facetype(h: int, w: int) -> NDArray[np.int32]:
    """Generate a 2D equirectangular segmentation image for each facetype.

    The generated segmentation image has lookup:

    * 0 - front
    * 1 - right
    * 2 - back
    * 3 - left
    * 4 - up
    * 5 - down

    See ``Face``.

    Example:

        >>> equirect_facetype(8, 12)
            array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=int32)

    Parameters
    ----------
    h: int
        Desired output height.
    w: int
        Desired output width. Must be a multiple of 4.

    Returns
    -------
    ndarray
        2D numpy equirectangular segmentation image for the 6 face types.
    """
    if w % 4:
        raise ValueError(f"w must be a multiple of 4. Got {w}.")

    # Create the pattern [2,3,3,0,0,1,1,2]
    w4 = w // 4
    w8 = w // 8
    h3 = h // 3
    tp = np.empty((h, w), dtype=np.int32)
    tp[:, :w8] = 2
    tp[:, w8 : w8 + w4] = 3
    tp[:, w8 + w4 : w8 + 2 * w4] = 0
    tp[:, w8 + 2 * w4 : w8 + 3 * w4] = 1
    tp[:, w8 + 3 * w4 :] = 2

    # Prepare ceil mask
    idx = np.linspace(-np.pi, np.pi, w4) / 4
    idx = np.round(h / 2 - np.arctan(np.cos(idx)) * h / np.pi).astype(np.int32)
    # It'll never go past a third of the image, so only process that for optimization
    mask = np.empty((h3, w4), np.bool_)
    row_idx = np.arange(h3, dtype=np.int32)[:, None]
    np.less(row_idx, idx[None], out=mask)

    flip_mask = np.flip(mask, 0)
    tp[:h3, :w8][mask[:, w8:]] = Face.UP
    tp[-h3:, :w8][flip_mask[:, w8:]] = Face.DOWN
    for i in range(3):
        s = slice_chunk(i, w4, w8)
        tp[:h3, s][mask] = Face.UP
        tp[-h3:, s][flip_mask] = Face.DOWN
    remainder = w - s.stop # pyright: ignore[reportPossiblyUnboundVariable]
    tp[:h3, s.stop :][mask[:, :remainder]] = Face.UP  # pyright: ignore[reportPossiblyUnboundVariable]
    tp[-h3:, s.stop :][flip_mask[:, :remainder]] = Face.DOWN  # pyright: ignore[reportPossiblyUnboundVariable]

    # Since we are using lru_cache, we want the return value to be immutable.
    # tp.setflags(write=False) # Disabled for now

    return tp


def xyzpers(
    h_fov: float, v_fov: float, u: float, v: float, out_hw: tuple[int, int], in_rot: float
) -> NDArray[np.float32]:
    """
    Generate XYZ coordinates for a perspective view grid.

    Parameters
    ----------
    h_fov : float
        Horizontal field of view in radians.
    v_fov : float
        Vertical field of view in radians.
    u : float
        Azimuth angle (rotation around Y axis) in radians.
    v : float
        Elevation angle (rotation around X axis) in radians.
    out_hw : tuple[int, int]
        Output height and width (H, W) of the perspective grid.
    in_rot : float
        In-plane rotation (roll) angle in radians around the view direction (Z axis).

    Returns
    -------
    NDArray[np.float32]
        (H, W, 3) array of XYZ coordinates.
    """
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1) # Standard perspective grid (z=1 initially)

    # Define extrinsic rotations (around fixed world axes, applied in order Y then X)
    # These rotate the *camera* frame relative to the world.
    # To rotate the *points*, we apply the inverse transforms in reverse order.
    Rx = rotation_matrix(v, Dim.X) # Rotation for elevation (around world X)
    Ry = rotation_matrix(u, Dim.Y) # Rotation for azimuth (around world Y)

    # Define intrinsic rotation (roll, around the camera's Z axis *after* extrinsic rotations)
    # Find the final Z-axis direction after applying Ry and Rx to the initial Z-axis [0,0,1]
    initial_z_axis = np.array([0.0, 0.0, 1.0])
    rotated_z_axis = initial_z_axis @ Ry.T @ Rx.T # Order matters for extrinsic composition
    Ri = rotation_matrix(in_rot, rotated_z_axis) # Rotation around the final view direction

    # Apply rotations to the grid points: Intrinsic (Ri), then Extrinsic (Rx, Ry)
    # Transformation: World -> Camera = (Intrinsic Roll @ Extrinsic Elev @ Extrinsic Azim)
    # To transform points from Camera grid coords to World coords:
    # World_coords = Camera_coords @ Ri.T @ Rx.T @ Ry.T
    out = out @ Ri.T # Apply inverse intrinsic rotation first
    out = out @ Rx.T # Then inverse extrinsic X rotation
    out = out @ Ry.T # Then inverse extrinsic Y rotation

    return out.astype(np.float32)


def xyz2uv(xyz: NDArray[DType]) -> tuple[NDArray[DType], NDArray[DType]]:
    """Transform cartesian (x,y,z) to spherical(r, u, v), and only outputs (u, v).

    Parameters
    ----------
    xyz: ndarray
        An array object in shape of [..., 3].

    Returns
    -------
    u, v: tuple[ndarray, ndarray]
        Arrays for longitude (u) and latitude (v) in radians.
        u is in [-pi, pi]
        v is in [-pi/2, pi/2]

    Notes
    -----
    Assumes standard coordinate system: +x right, +y up, +z out of screen.
    u (azimuth/longitude) is angle in xz-plane from +z axis (0) towards +x axis (pi/2).
    v (elevation/latitude) is angle from xz-plane towards +y axis.
    """
    # Use slicing to keep dimensions without copying if possible
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]
    u = np.arctan2(x, z)  # Angle in xz plane: 0 along +z, pi/2 along +x, pi along -z, -pi/2 along -x
    c = np.hypot(x, z)    # Distance in xz plane
    v = np.arctan2(y, c)  # Angle from xz plane towards y
    return u, v


def uv2unitxyz(uv: NDArray[DType]) -> NDArray[DType]:
    """Convert spherical coordinates (u, v) to unit Cartesian coordinates (x, y, z).

    Parameters
    ----------
    uv : NDArray[DType]
        Array of shape [..., 2] containing (u, v) coordinates in radians.

    Returns
    -------
    NDArray[DType]
        Array of shape [..., 3] containing (x, y, z) coordinates on the unit sphere.
    """
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)
    return np.concatenate([x, y, z], axis=-1, dtype=uv.dtype)


def uv2coor(u: NDArray[DType], v: NDArray[DType], h: int, w: int) -> tuple[NDArray[DType], NDArray[DType]]:
    """Transform spherical(u, v) in radians to equirectangular pixel coordinates (x, y).

    Parameters
    ----------
    u: ndarray
        Longitude/azimuth in radians, range [-pi, pi].
    v: ndarray
        Latitude/elevation in radians, range [-pi/2, pi/2].
    h: int
        Height of the target equirectangular image.
    w: int
        Width of the target equirectangular image.

    Returns
    -------
    coor_x, coor_y: tuple[ndarray, ndarray]
        Corresponding pixel coordinates (horizontal, vertical).
        coor_x range is roughly [-0.5, w-0.5].
        coor_y range is roughly [-0.5, h-0.5].

    Notes
    -----
    Maps u=-pi to x=-0.5, u=pi to x=w-0.5.
    Maps v=pi/2 to y=-0.5, v=-pi/2 to y=h-0.5.
    Pixel centers are at integer coordinates (0,0), (1,0), ...
    """
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5  # Map [-pi, pi] to [-0.5, w-0.5]
    coor_y = (-v / np.pi + 0.5) * h - 0.5  # Map [pi/2, -pi/2] to [-0.5, h-0.5]
    return coor_x, coor_y


def coor2uv(coorxy: NDArray[DType], h: int, w: int) -> NDArray[DType]:
    """Transform equirectangular pixel coordinates (x, y) to spherical(u, v) in radians.

    Parameters
    ----------
    coorxy : NDArray[DType]
        Array of shape [..., 2] containing (x, y) pixel coordinates.
    h : int
        Height of the equirectangular image.
    w : int
        Width of the equirectangular image.

    Returns
    -------
    NDArray[DType]
        Array of shape [..., 2] containing (u, v) coordinates in radians.
    """
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi  # Map x to u [-pi, pi]
    v = -((coor_y + 0.5) / h - 0.5) * np.pi  # Map y to v [pi/2, -pi/2]
    return np.concatenate([u, v], axis=-1, dtype=coorxy.dtype)


class EquirecSampler:
    """Samples an equirectangular image based on provided coordinates."""
    def __init__(
        self,
        coor_x: NDArray,
        coor_y: NDArray,
        order: int,
    ):
        """
        Initializes the sampler.

        Args:
            coor_x (NDArray): The x-coordinates (horizontal) to sample from the equirectangular image.
                              Shape should match the desired output shape. Values correspond to pixel
                              coordinates in the source equirectangular image.
            coor_y (NDArray): The y-coordinates (vertical) to sample. Shape must match coor_x.
            order (int): The order of spline interpolation (0=nearest, 1=linear, 3=cubic, etc.).
                         See scipy.ndimage.map_coordinates.
        """
        # Add 1 to the coordinates to compensate for the 1 pixel padding later applied.
        coor_x_pad = coor_x + 1.0  # Not done inplace on purpose.
        coor_y_pad = coor_y + 1.0  # Not done inplace on purpose.

        self._shape = coor_x.shape # Store original shape of coordinates (output shape)

        if cv2 and order in (0, 1, 3):
            self._use_cv2 = True
            if order == 0:
                self._order = cv2.INTER_NEAREST
                nninterpolation = True
            elif order == 1:
                self._order = cv2.INTER_LINEAR
                nninterpolation = False
            elif order == 3:
                self._order = cv2.INTER_CUBIC
                nninterpolation = False
            else:
                # Should not happen based on check 'order in (0, 1, 3)'
                raise NotImplementedError("Internal error: Unsupported CV2 order.")

            # OpenCV remap expects maps in float32
            self._map_x, self._map_y = cv2.convertMaps(
                coor_x_pad.astype(np.float32),
                coor_y_pad.astype(np.float32),
                cv2.CV_16SC2, # Output type for fixed-point maps
                nninterpolation=nninterpolation,
            )
        else:
            self._use_cv2 = False
            # Store coordinates for map_coordinates (expects y, x order)
            # map_coordinates needs float coordinates.
            self._coords_for_scipy = np.stack(
                (coor_y_pad, coor_x_pad), axis=0
            ).astype(np.float64) # Use float64 for potentially better precision
            self._order = order

    def __call__(self, img: NDArray[DType]) -> NDArray[DType]:
        """Samples the input equirectangular image.

        Args:
            img (NDArray): The source equirectangular image. Can be 2D (H, W) or 3D (H, W, C).

        Returns:
            NDArray: The sampled image, with shape matching the coordinates provided during init.
                     If input was 3D, output is also 3D with the same number of channels.
        """
        if img.ndim not in (2, 3):
             raise ValueError(f"Input image must be 2D or 3D, got {img.ndim} dimensions.")

        source_dtype = img.dtype
        # Scipy/CV2 often work best with float32/float64 or uint8
        if source_dtype == np.float16:
            img_proc = img.astype(np.float32)
        elif source_dtype not in (np.uint8, np.float32, np.float64):
            # Promote other types (like int) to float32 for interpolation
            # print(f"Warning: EquirecSampler promoting input dtype {source_dtype} to float32 for interpolation.")
            img_proc = img.astype(np.float32)
        else:
            img_proc = img

        # Pad the image
        padded = self._pad(img_proc)

        # Determine if image has channels
        has_channels = padded.ndim == 3 and padded.shape[-1] > 1 # Handle case of (H,W,1)
        if not has_channels and padded.ndim == 2:
            # Temporarily add channel dim for consistent processing
             padded = padded[..., None]
             num_channels = 1
        elif padded.ndim == 3:
             num_channels = padded.shape[2]
        else:
             # Should not happen based on input check and padding logic
             raise ValueError("Internal error: Unexpected image dimensions after processing.")


        channels = []

        for i in range(num_channels):
            channel_data = padded[..., i]
            if self._use_cv2:
                # cv2.remap needs float32 or uint8 input
                if channel_data.dtype not in (np.uint8, np.float32):
                     channel_data = channel_data.astype(np.float32)

                # Perform remapping
                sampled_channel = cv2.remap(
                    channel_data,
                    self._map_x,
                    self._map_y,
                    interpolation=self._order,
                    borderMode=cv2.BORDER_REPLICATE # Should sample within padded area due to manual wrap padding
                )
            else:
                # map_coordinates expects data, coords=(y,x,...), order
                # coords shape should be (ndim, ...) where ... is output shape
                sampled_channel = map_coordinates(
                    channel_data,
                    self._coords_for_scipy, # Already has shape (2, H_out, W_out)
                    order=self._order,
                    mode='wrap', # Use wrap for equirectangular horizontal continuity (already handled by padding?)
                                 # 'nearest' might be safer given manual padding. Test needed. Let's use 'nearest' for padded data.
                                 # Correction: 'wrap' is still needed for the horizontal axis logic in scipy.
                                 #             'edge' is implicitly handled for vertical by _pad.
                    cval=0.0,    # Value for points outside boundary (shouldn't happen with padding)
                    prefilter=True if self._order > 1 else False # Prefilter for higher orders
                )
            channels.append(sampled_channel)

        # Stack channels back if needed
        if num_channels > 1:
            out = np.stack(channels, axis=-1)
        else:
            out = channels[0] # Single channel, shape is already (H_out, W_out)

        # Ensure output shape matches coordinates shape exactly
        if out.shape[:2] != self._shape:
             print(f"Warning: Output shape {out.shape[:2]} differs slightly from coordinate shape {self._shape}. Resizing.")
             # This might indicate an issue elsewhere, but resize as a fallback
             if cv2:
                 # Determine target size correctly (W, H for cv2.resize)
                 target_size = (self._shape[1], self._shape[0])
                 # Choose suitable interpolation for resizing itself
                 resize_interp = self._order if self._use_cv2 else cv2.INTER_LINEAR
                 if out.ndim == 3 and out.shape[-1] == 1: # Handle single channel 3D array
                    out = cv2.resize(out[...,0], target_size, interpolation=resize_interp)[...,None]
                 else:
                    out = cv2.resize(out, target_size, interpolation=resize_interp)
             else:
                 # Fallback numpy resize is complex, maybe error instead?
                 print("Error: Cannot easily resize without OpenCV. Check coordinate generation.")
                 # Attempt simple slicing/padding - risky
                 # out = out[:self._shape[0], :self._shape[1]] # Simplistic crop


        # Cast back to original dtype if needed
        if out.dtype != source_dtype:
            if np.issubdtype(source_dtype, np.integer):
                 # Clamp and round before casting to integer
                 out = np.clip(np.round(out), np.iinfo(source_dtype).min, np.iinfo(source_dtype).max)
            out = out.astype(source_dtype)

        return out

    def _pad(self, img: NDArray) -> NDArray:
        """Adds 1 pixel padding for interpolation, handling equirectangular wrapping.

        Pads manually: 'wrap' for horizontal (W), 'edge' for vertical (H).
        """
        h, w = img.shape[:2]
        num_dims = img.ndim

        # Create padded array structure
        pad_shape_list = list(img.shape)
        pad_shape_list[0] += 2
        pad_shape_list[1] += 2
        padded = np.empty(tuple(pad_shape_list), dtype=img.dtype)

        # --- Fill Center ---
        if num_dims == 2:
            padded[1:-1, 1:-1] = img
        else: # 3D
            padded[1:-1, 1:-1, ...] = img

        # --- Fill Horizontal Edges (Wrap) ---
        # Left edge (copies right edge of original)
        padded[1:-1, 0, ...] = img[:, -1, ...] if num_dims == 3 else img[:, -1]
        # Right edge (copies left edge of original)
        padded[1:-1, -1, ...] = img[:, 0, ...] if num_dims == 3 else img[:, 0]

        # --- Fill Vertical Edges (Edge/Replicate) ---
        # Top edge (copies top edge of original, including newly wrapped sides)
        padded[0, :, ...] = padded[1, :, ...]
        # Bottom edge (copies bottom edge of original, including newly wrapped sides)
        padded[-1, :, ...] = padded[-2, :, ...]

        # Corners are filled implicitly by the vertical edge replication after horizontal wrap

        expected_shape = tuple(pad_shape_list)
        if padded.shape != expected_shape:
             # This check should ideally never fail with this manual padding
             print(f"INTERNAL PADDING ERROR: Got shape {padded.shape}, expected {expected_shape}")

        return padded


    @classmethod
    # @lru_cache(_CACHE_SIZE) # Cache disabled
    def from_cubemap(cls, face_w: int, h: int, w: int, order: int):
        """Construct a EquirecSampler to sample equirectangular for cubemap faces.

        Parameters
        ----------
        face_w: int
            Length of each face of the output cubemap (determines output shape).
        h: int
            Height of input equirec image.
        w: int
            Width of input equirec image.
        order: int
            The order of the spline interpolation.
        """
        # Get XYZ coords on the unit cube faces
        xyz = xyzcube(face_w) # Shape (face_w, face_w * 6, 3)

        # Convert XYZ to UV angles (longitude, latitude)
        u, v = xyz2uv(xyz) # Shapes (face_w, face_w * 6, 1) each

        # Convert UV angles to equirectangular pixel coordinates
        coor_x, coor_y = uv2coor(u, v, h, w) # Shapes (face_w, face_w * 6, 1) each

        # Squeeze the last dimension as Sampler expects 2D coordinate maps
        return cls(coor_x.squeeze(-1), coor_y.squeeze(-1), order=order)

    @classmethod
    # @lru_cache(_CACHE_SIZE) # Cache disabled
    def from_perspective(
        cls, h_fov_rad: float, v_fov_rad: float, u_rad: float, v_rad: float, roll_rad: float,
        in_h: int, in_w: int, out_h: int, out_w: int, order: int
    ):
        """Construct a EquirecSampler to sample equirectangular for a perspective view.

        Parameters are expected in radians and image dimensions.
        """
        # Get XYZ coords for the perspective grid
        # xyzpers uses radians for fov, angles and image H, W
        xyz = xyzpers(h_fov_rad, v_fov_rad, u_rad, v_rad, (out_h, out_w), roll_rad) # Shape (out_h, out_w, 3)

        # Convert XYZ to UV angles (longitude, latitude) in radians
        u, v = xyz2uv(xyz) # Shapes (out_h, out_w, 1) each

        # Convert UV angles to equirectangular pixel coordinates (x, y)
        # uv2coor uses radians for u,v and image H, W
        coor_x, coor_y = uv2coor(u, v, in_h, in_w) # Shapes (out_h, out_w, 1) each

        # Squeeze the last dimension as Sampler expects 2D coordinate maps
        return cls(coor_x.squeeze(-1), coor_y.squeeze(-1), order=order)


class CubeFaceSampler:
    """Samples pixels from cube faces to create an output image (e.g., equirectangular)."""
    # Implementation note: This class was in the original utils.py but isn't strictly
    # required for the e2c (equirectangular-to-cubemap) conversion.
    # It would be used for the inverse (cubemap-to-equirectangular).
    # Including it here for completeness of the original utils.py content.
    def __init__(
        self,
        tp: NDArray[np.int32],
        coor_x: NDArray[np.float32],
        coor_y: NDArray[np.float32],
        order: int,
        face_w: int, # Expected width/height of input cube faces
    ):
        """Initializes sampler and performs pre-computations.

        Parameters
        ----------
        tp: numpy.ndarray
            (H_out, W_out) facetype map indicating which face (0-5) each output pixel comes from.
        coor_x: numpy.ndarray
            (H_out, W_out) X coordinates to sample *within* the respective cube face. Range [0, face_w].
        coor_y: numpy.ndarray
            (H_out, W_out) Y coordinates to sample *within* the respective cube face. Range [0, face_w].
        order: int
            The order of the spline interpolation.
        face_w: int
            The width/height of the source cube faces.
        """
        # Add 1 to compensate for 1-pixel-surround padding applied later.
        coor_x_pad = coor_x + 1.0
        coor_y_pad = coor_y + 1.0

        self._tp = tp # Face type map (H_out, W_out)
        self._face_w = face_w # Expected input face size
        self._out_shape = tp.shape # Shape of the output image

        if cv2 and order in (0, 1, 3):
            self._use_cv2 = True
            if order == 0:
                self._order = cv2.INTER_NEAREST
                nninterpolation = True
            elif order == 1:
                self._order = cv2.INTER_LINEAR
                nninterpolation = False
            elif order == 3:
                self._order = cv2.INTER_CUBIC
                nninterpolation = False
            else:
                raise NotImplementedError("Internal error: Unsupported CV2 order.")

            # For OpenCV, we need to map coordinates to a vertically stacked image of padded faces.
            # The effective Y coordinate becomes face_index * (face_h + 2) + coor_y_pad
            # We assume face_h = face_w here. Pad is 1 on each side, so +2.
            effective_coor_y = coor_y_pad + tp.astype(np.float32) * (face_w + 2)

            # CV_16SC2 expects float32 inputs for map generation
            self._map_x, self._map_y = cv2.convertMaps(
                coor_x_pad.astype(np.float32),
                effective_coor_y.astype(np.float32),
                cv2.CV_16SC2,
                nninterpolation=nninterpolation,
            )
        else:
            self._use_cv2 = False
            # Store coordinates for map_coordinates, expects (face_idx, y, x)
            self._coords_for_scipy = np.stack(
                (tp, coor_y_pad, coor_x_pad), axis=0 # Shape (3, H_out, W_out)
            ).astype(np.float64) # Use float64 for precision
            self._order = order

    def __call__(self, cube_faces: NDArray[DType]) -> NDArray[DType]:
        """Sample cube faces to generate the output image.

        Parameters
        ----------
        cube_faces: numpy.ndarray
             Input cube faces, expected shape (6, S, S, C) or (6, S, S) where S = face_w.

        Returns
        -------
        numpy.ndarray
            (H_out, W_out, C) or (H_out, W_out) Sampled image.
        """
        if not (cube_faces.ndim == 3 or cube_faces.ndim == 4):
             raise ValueError(f"Input cube_faces must be 3D (6,S,S) or 4D (6,S,S,C), got {cube_faces.ndim} dims.")
        if cube_faces.shape[0] != 6:
             raise ValueError(f"Input cube_faces must have 6 faces in the first dimension, got {cube_faces.shape[0]}.")
        if cube_faces.shape[1] != self._face_w or cube_faces.shape[2] != self._face_w:
             raise ValueError(f"Input cube face size ({cube_faces.shape[1]}x{cube_faces.shape[2]}) "
                              f"doesn't match expected size {self._face_w}x{self._face_w}.")

        source_dtype = cube_faces.dtype
        # Promote types if necessary for interpolation libraries
        if source_dtype == np.float16:
            cube_proc = cube_faces.astype(np.float32)
        elif source_dtype not in (np.uint8, np.float32, np.float64):
            # print(f"Warning: CubeFaceSampler promoting input dtype {source_dtype} to float32 for interpolation.")
            cube_proc = cube_faces.astype(np.float32)
        else:
            cube_proc = cube_faces

        # Pad each face
        padded = self._pad(cube_proc) # Shape (6, S+2, S+2, C) or (6, S+2, S+2)

        # Determine if image has channels
        has_channels = padded.ndim == 4 and padded.shape[-1] > 1
        if not has_channels and padded.ndim == 3:
             padded = padded[..., None] # Add channel dim
             num_channels = 1
        elif padded.ndim == 4:
             num_channels = padded.shape[3]
        else:
             raise ValueError("Internal error: Unexpected cube dimensions after processing.")

        channels = []

        for i in range(num_channels):
            channel_data = padded[..., i] # Shape (6, S+2, S+2)
            if self._use_cv2:
                 # Vertically stack the padded faces for cv2.remap
                 # Shape (6, S+2, S+2) -> (6*(S+2), S+2)
                 v_stack = channel_data.reshape(-1, self._face_w + 2)

                 # Ensure data type is compatible with cv2.remap
                 if v_stack.dtype not in (np.uint8, np.float32):
                     v_stack = v_stack.astype(np.float32)

                 sampled_channel = cv2.remap(
                    v_stack,
                    self._map_x,
                    self._map_y,
                    interpolation=self._order,
                    borderMode=cv2.BORDER_CONSTANT, # Use constant border, padding handles connections
                    borderValue=0 # Value if somehow sampling outside padded stack
                )
            else:
                # map_coordinates takes data[face, y, x] and coords[dim, out_coords]
                sampled_channel = map_coordinates(
                    channel_data, # Shape (6, S+2, S+2)
                    self._coords_for_scipy, # Shape (3, H_out, W_out) corresponding to (face, y, x)
                    order=[0, self._order, self._order], # Nearest for face index, specified order for coords
                    mode='nearest', # Sample within padded area, 'nearest' handles boundaries.
                    cval=0.0,
                    prefilter=True if self._order > 1 else False
                ) # Output shape (H_out, W_out)
            channels.append(sampled_channel)

        # Stack channels back
        if num_channels > 1:
            out = np.stack(channels, axis=-1)
        else:
            out = channels[0]

        # Cast back to original dtype
        if out.dtype != source_dtype:
            if np.issubdtype(source_dtype, np.integer):
                 out = np.clip(np.round(out), np.iinfo(source_dtype).min, np.iinfo(source_dtype).max)
            out = out.astype(source_dtype)

        # Ensure output shape matches expected output shape
        if out.shape[:2] != self._out_shape:
             print(f"Warning: CubeFaceSampler output shape {out.shape[:2]} differs from expected {self._out_shape}. Resizing.")
             if cv2:
                 target_size = (self._out_shape[1], self._out_shape[0])
                 resize_interp = self._order if self._use_cv2 else cv2.INTER_LINEAR
                 if out.ndim == 3 and out.shape[-1] == 1:
                     out = cv2.resize(out[...,0], target_size, interpolation=resize_interp)[...,None]
                 else:
                     out = cv2.resize(out, target_size, interpolation=resize_interp)

             else:
                 print("Error: Cannot easily resize CubeFaceSampler output without OpenCV.")


        return out # pyright: ignore[reportReturnType]


    def _pad(self, cube_faces: NDArray[DType]) -> NDArray[DType]:
        """Adds 1 pixel padding around each cube face, handling adjacent face connections."""
        S = self._face_w
        has_channels = cube_faces.ndim == 4
        pad_width = ((0, 0), (1, 1), (1, 1), (0,0)) if has_channels else ((0,0), (1,1), (1,1))
        # Start with edge padding as a base - corners and edges will be overwritten
        padded = np.pad(cube_faces, pad_width, mode='edge')

        F, R, B, L, U, D = Face.FRONT, Face.RIGHT, Face.BACK, Face.LEFT, Face.UP, Face.DOWN
        # Slice indices within the padded (S+2, S+2) face:
        PAD_T, PAD_B, PAD_L, PAD_R = 0, S + 1, 0, S + 1  # Indices of the padding rows/cols
        CORE_T, CORE_B, CORE_L, CORE_R = 1, S + 1, 1, S + 1 # Indices for slicing the core data region (1 to S+1 exclusive end)
        CORE_SLICE = slice(CORE_T, CORE_B) # Slice(1, S+1)

        # --- Fill Edges using data from adjacent faces' edges (non-padded indices) ---
        # Note: Adjust indexing based on relative orientation (some need flips)
        # Data comes from the original `cube_faces` array (non-padded)

        # Front face neighbours
        padded[F, PAD_T, CORE_SLICE, ...]    = cube_faces[U, -1, :, ...]          # Top edge from Up's bottom edge
        padded[F, PAD_B, CORE_SLICE, ...]    = cube_faces[D, 0, :, ...]           # Bottom edge from Down's top edge
        padded[F, CORE_SLICE, PAD_L, ...]    = cube_faces[L, :, -1, ...]          # Left edge from Left's right edge
        padded[F, CORE_SLICE, PAD_R, ...]    = cube_faces[R, :, 0, ...]           # Right edge from Right's left edge

        # Right face neighbours
        padded[R, PAD_T, CORE_SLICE, ...]    = np.flipud(cube_faces[U, :, -1, ...]) # Top edge from Up's right edge (flippedud)
        padded[R, PAD_B, CORE_SLICE, ...]    = np.flipud(cube_faces[D, :, -1, ...]) # Bottom edge from Down's right edge (flippedud)
        padded[R, CORE_SLICE, PAD_L, ...]    = cube_faces[F, :, -1, ...]          # Left edge from Front's right edge
        padded[R, CORE_SLICE, PAD_R, ...]    = cube_faces[B, :, 0, ...]           # Right edge from Back's left edge

        # Back face neighbours
        padded[B, PAD_T, CORE_SLICE, ...]    = np.fliplr(cube_faces[U, 0, :, ...])  # Top edge from Up's top edge (flippedlr)
        padded[B, PAD_B, CORE_SLICE, ...]    = np.fliplr(cube_faces[D, -1, :, ...]) # Bottom edge from Down's bottom edge (flippedlr)
        padded[B, CORE_SLICE, PAD_L, ...]    = cube_faces[R, :, -1, ...]          # Left edge from Right's right edge
        padded[B, CORE_SLICE, PAD_R, ...]    = cube_faces[L, :, 0, ...]           # Right edge from Left's left edge

        # Left face neighbours
        padded[L, PAD_T, CORE_SLICE, ...]    = np.flipud(cube_faces[U, :, 0, ...])  # Top edge from Up's left edge (flippedud)
        padded[L, PAD_B, CORE_SLICE, ...]    = np.flipud(cube_faces[D, :, 0, ...])  # Bottom edge from Down's left edge (flippedud)
        padded[L, CORE_SLICE, PAD_L, ...]    = cube_faces[B, :, -1, ...]          # Left edge from Back's right edge
        padded[L, CORE_SLICE, PAD_R, ...]    = cube_faces[F, :, 0, ...]           # Right edge from Front's left edge

        # Up face neighbours
        padded[U, PAD_T, CORE_SLICE, ...]    = cube_faces[B, 0, ::-1, ...]        # Top edge from Back's top edge (flippedlr)
        padded[U, PAD_B, CORE_SLICE, ...]    = cube_faces[F, 0, :, ...]           # Bottom edge from Front's top edge
        padded[U, CORE_SLICE, PAD_L, ...]    = cube_faces[L, 0, :, ...]           # Left edge from Left's top edge (needs flipud?) No, Left T corresponds to Up L.
        padded[U, CORE_SLICE, PAD_R, ...]    = cube_faces[R, 0, :, ...]           # Right edge from Right's top edge (needs flipud?) No, Right T corresponds to Up R.

        # Down face neighbours
        padded[D, PAD_T, CORE_SLICE, ...]    = cube_faces[F, -1, :, ...]          # Top edge from Front's bottom edge
        padded[D, PAD_B, CORE_SLICE, ...]    = cube_faces[B, -1, ::-1, ...]        # Bottom edge from Back's bottom edge (flippedlr)
        padded[D, CORE_SLICE, PAD_L, ...]    = cube_faces[L, -1, :, ...]          # Left edge from Left's bottom edge (needs flipud?) No.
        padded[D, CORE_SLICE, PAD_R, ...]    = cube_faces[R, -1, :, ...]          # Right edge from Right's bottom edge (needs flipud?) No.


        # --- Fill Corners (using adjacent edge values *after* they've been filled) ---
        # This simplified approach replicates the corner pixel from the most relevant already-padded neighbor edge pixel.
        # More complex corner handling might average or use diagonal neighbours if needed, but this is often sufficient.
        padded[:, PAD_T, PAD_L, ...] = padded[:, PAD_T, CORE_T, ...]       # Top-left corner from top edge's first pixel
        padded[:, PAD_T, PAD_R, ...] = padded[:, PAD_T, CORE_R-1, ...]     # Top-right corner from top edge's last pixel
        padded[:, PAD_B, PAD_L, ...] = padded[:, PAD_B, CORE_T, ...]       # Bottom-left corner from bottom edge's first pixel
        padded[:, PAD_B, PAD_R, ...] = padded[:, PAD_B, CORE_R-1, ...]     # Bottom-right corner from bottom edge's last pixel


        return padded

    @classmethod
    # @lru_cache(_CACHE_SIZE) # Cache disabled
    def from_equirec(cls, face_w: int, h: int, w: int, order: int):
        """Construct a CubeFaceSampler to sample cube faces for an equirectangular output.

        Parameters
        ----------
        face_w: int
            Length of each face of the input cubemap.
        h: int
            Output equirectangular image height.
        w: int
            Output equirectangular image width. Must be multiple of 4.
        order: int
            The order of the spline interpolation.
        """
        # Get UV grid for the output equirectangular image
        u, v = equirect_uvgrid(h, w) # u [-pi,pi], v [pi/2,-pi/2], shapes (h, w)

        # Get face id (0-5) for each pixel in the output equirectangular image
        tp = equirect_facetype(h, w) # Shape (h, w)

        # Calculate the coordinates (x, y) within the *source* cube face
        # corresponding to each (u, v) angle. Output range is [0, face_w].
        coor_x = np.zeros((h, w), dtype=np.float32)
        coor_y = np.zeros((h, w), dtype=np.float32)
        face_w_float = float(face_w)
        scale = face_w_float / 2.0

        # Project view vector (defined by u, v) onto the corresponding cube face plane
        # Formulas derived from projecting (x,y,z) = (sin(u)cos(v), sin(v), cos(u)cos(v))
        # onto the plane of the face (e.g., z=0.5 for Front) and scaling/offsetting.

        # Common terms
        cos_v = np.cos(v)
        sin_v = np.sin(v)
        cos_u = np.cos(u)
        sin_u = np.sin(u)

        # Calculate coords for ALL points first, then use mask to assign.
        # This is often faster than masking intermediate calculations.

        # Front (z=0.5): x = 0.5 * tan(u), y = 0.5 * tan(v) / cos(u)
        # Face Coords: Xf = x + 0.5, Yf = -y + 0.5 => Xf = 0.5*tan(u)+0.5, Yf = -0.5*tan(v)/cos(u)+0.5
        # Avoid division by zero/inf at edges using safe division or clipping intermediates
        safe_cos_u = np.where(np.abs(cos_u) < 1e-6, 1e-6, cos_u)
        fx = scale * (np.tan(u) + 1.0)
        fy = scale * (-np.tan(v) / safe_cos_u + 1.0)
        mask = (tp == Face.FRONT)
        coor_x[mask] = fx[mask]
        coor_y[mask] = fy[mask]

        # Right (x=0.5): z = 0.5 * cot(u), y = 0.5 * tan(v) / sin(u)
        # Face Coords: Xr = -z + 0.5, Yr = -y + 0.5 => Xr = -0.5*cot(u)+0.5, Yr = -0.5*tan(v)/sin(u)+0.5
        safe_sin_u = np.where(np.abs(sin_u) < 1e-6, 1e-6, sin_u)
        rx = scale * (-1.0 / np.tan(u) + 1.0) if np.all(np.abs(np.tan(u))>1e-6) else scale # Approx cot = 1/tan
        ry = scale * (-np.tan(v) / safe_sin_u + 1.0)
        mask = (tp == Face.RIGHT)
        coor_x[mask] = rx[mask]
        coor_y[mask] = ry[mask]

        # Back (z=-0.5): x = -0.5 * tan(u), y = -0.5 * tan(v) / cos(u)
        # Face Coords: Xb = -x + 0.5, Yb = -y + 0.5 => Xb = 0.5*tan(u)+0.5, Yb = 0.5*tan(v)/cos(u)+0.5
        bx = scale * (np.tan(u) + 1.0)
        by = scale * (np.tan(v) / safe_cos_u + 1.0)
        mask = (tp == Face.BACK)
        coor_x[mask] = bx[mask]
        coor_y[mask] = by[mask]

        # Left (x=-0.5): z = -0.5 * cot(u), y = -0.5 * tan(v) / sin(u)
        # Face Coords: Xl = z + 0.5, Yl = -y + 0.5 => Xl = -0.5*cot(u)+0.5, Yl = 0.5*tan(v)/sin(u)+0.5
        lx = scale * (-1.0 / np.tan(u) + 1.0) if np.all(np.abs(np.tan(u))>1e-6) else scale
        ly = scale * (np.tan(v) / safe_sin_u + 1.0)
        mask = (tp == Face.LEFT)
        coor_x[mask] = lx[mask]
        coor_y[mask] = ly[mask]

        # Up (y=0.5): x = 0.5 * tan(u) * cos(v) / sin(v), z = 0.5 * cos(v) / sin(v)
        # Face Coords: Xu = x + 0.5, Yu = z + 0.5 => Xu = 0.5*tan(u)/tan(v)+0.5, Yu = 0.5/tan(v)+0.5
        safe_sin_v = np.where(np.abs(sin_v) < 1e-6, 1e-6, sin_v) # Avoid pole issue
        safe_tan_v = safe_sin_v / np.where(np.abs(cos_v) < 1e-6, 1e-6, cos_v)
        ux = scale * (np.tan(u) / safe_tan_v + 1.0)
        uy = scale * (1.0 / safe_tan_v + 1.0)
        mask = (tp == Face.UP)
        coor_x[mask] = ux[mask]
        coor_y[mask] = uy[mask]

        # Down (y=-0.5): x = -0.5 * tan(u) * cos(v) / sin(v), z = -0.5 * cos(v) / sin(v)
        # Face Coords: Xd = x + 0.5, Yd = -z + 0.5 => Xd = -0.5*tan(u)/tan(v)+0.5, Yd = 0.5/tan(v)+0.5
        dx = scale * (-np.tan(u) / safe_tan_v + 1.0)
        dy = scale * (1.0 / safe_tan_v + 1.0) # Note v is negative here, so tan(v) is negative
        mask = (tp == Face.DOWN)
        coor_x[mask] = dx[mask]
        coor_y[mask] = dy[mask]

        # Scale coords from [0, 1] range (relative to face size) to [0, face_w] pixel coords
        # This was already done via `scale = face_w / 2.0` multiplier. Need to ensure result is pixel coord.
        # Let's re-check the formulas. The range should be [0, face_w].
        # Example: Front face, u=0, v=0 -> tan(u)=0, tan(v)=0 -> fx=scale*1, fy=scale*1 => center (face_w/2, face_w/2) - CORRECT.
        # Example: Front face, u=pi/4, v=0 -> tan(u)=1, tan(v)=0 -> fx=scale*2, fy=scale*1 => (face_w, face_w/2) - Right edge center - CORRECT.
        # Example: Front face, u=0, v=pi/4 -> tan(u)=0, tan(v)=1 -> fx=scale*1, fy=scale*(-1/1+1)=0 => (face_w/2, 0) - Top edge center - CORRECT.

        # Final clipping to ensure coords are within the valid pixel range [0, face_w-1] for sampling
        # Allowing slightly outside (e.g., -0.5 to face_w-0.5) is often handled by interpolation modes,
        # but explicit clipping is safer, especially near poles.
        # Since we pad later, the effective range for sampling is [-1, face_w].
        # Clipping to [0, face_w] should be fine.
        np.clip(coor_x, 0.0, face_w_float - 1e-4, out=coor_x) # Use small epsilon to avoid being exactly face_w
        np.clip(coor_y, 0.0, face_w_float - 1e-4, out=coor_y)

        return cls(tp, coor_x, coor_y, order, face_w)


# --- Cubemap Format Conversion Functions (from original utils.py) ---

def cube_h2list(cube_h: NDArray[DType]) -> list[NDArray[DType]]:
    """Split a horizontal cubemap strip image into a list of 6 faces."""
    h = cube_h.shape[0]
    w_total = cube_h.shape[1]
    if w_total % 6 != 0:
        raise ValueError(f"Cubemap width ({w_total}) must be divisible by 6.")
    face_w = w_total // 6
    # Allow non-square faces, check needed?
    # if h != face_w:
    #      print(f"Warning: cube_h2list input height {h} != width/6 {face_w}")

    num_dims = cube_h.ndim
    if num_dims == 2: # Grayscale HW
        return [cube_h[:, slice_chunk(i, face_w)] for i in range(6)]
    elif num_dims == 3: # Color HWC
        return [cube_h[:, slice_chunk(i, face_w), :] for i in range(6)]
    # Batched formats not explicitly supported here, assume HW or HWC input
    else:
        raise ValueError(f"Unsupported number of dimensions for cube_h: {num_dims}. Expected 2 (HW) or 3 (HWC).")


def cube_list2h(cube_list: list[NDArray[DType]]) -> NDArray[DType]:
    """Concatenate a list of 6 face images side-by-side into a horizontal strip."""
    if len(cube_list) != 6:
        raise ValueError(f"6 elements must be provided to construct a cube; got {len(cube_list)}.")

    # Basic shape/dtype checks
    first_face = cube_list[0]
    first_shape = first_face.shape
    first_dtype = first_face.dtype
    num_dims = first_face.ndim

    if num_dims not in (2, 3):
        raise ValueError(f"Faces must be 2D (HW) or 3D (HWC), got {num_dims} dimensions.")

    for i, face in enumerate(cube_list[1:], 1):
        if face.shape != first_shape:
            raise ValueError(
                f"Face {i}'s shape {face.shape} doesn't match the first face's shape {first_shape}."
            )
        if face.dtype != first_dtype:
            raise ValueError(
                f"Face {i}'s dtype {face.dtype} doesn't match the first face's dtype {first_dtype}."
            )

    # Concatenate along the width axis (axis 1 for HW and HWC)
    axis = 1
    return np.concatenate(cube_list, axis=axis, dtype=first_dtype)


def cube_h2dict(cube_h: NDArray[DType]) -> dict[str, NDArray[DType]]:
    """Convert a horizontal cubemap strip to a dictionary with keys F,R,B,L,U,D."""
    return dict(zip("FRBLUD", cube_h2list(cube_h)))


def cube_dict2list(cube_dict: dict[Any, NDArray[DType]], face_k: Optional[Sequence] = None) -> list[NDArray[DType]]:
    """Convert a dictionary of faces to a list in specified order."""
    face_k = face_k or "FRBLUD" # Default order
    if len(face_k) != 6:
        raise ValueError(f"6 face_k keys must be provided to construct a cube list; got {len(face_k)}.")
    if not all(k in cube_dict for k in face_k):
         missing = [k for k in face_k if k not in cube_dict]
         raise ValueError(f"Missing keys in cube_dict: {missing}")

    return [cube_dict[k] for k in face_k]


def cube_dict2h(cube_dict: dict[Any, NDArray[DType]], face_k: Optional[Sequence] = None) -> NDArray[DType]:
    """Convert a dictionary of faces to a horizontal strip using specified key order."""
    return cube_list2h(cube_dict2list(cube_dict, face_k))


def cube_h2dice(cube_h: NDArray[DType]) -> NDArray[DType]:
    """Convert a horizontal cubemap strip to the 'dice' format (cross layout)."""
    h = cube_h.shape[0]
    w_total = cube_h.shape[1]
    if w_total % 6 != 0:
        raise ValueError(f"Cubemap width ({w_total}) must be divisible by 6.")
    face_w = w_total // 6
    # Allow non-square
    # if h != face_w:
    #     print(f"Warning: cube_h2dice input height {h} != width/6 {face_w}")

    cube_list = cube_h2list(cube_h)
    # Assume standard F R B L U D order from cube_h2list
    F, R, B, L, U, D = cube_list

    out_h = h * 3
    out_w = face_w * 4
    channels = cube_h.shape[2] if cube_h.ndim == 3 else None

    if channels is not None:
        cube_dice = np.zeros((out_h, out_w, channels), dtype=cube_h.dtype)
    else:
        cube_dice = np.zeros((out_h, out_w), dtype=cube_h.dtype)

    # Define slices for dice layout (row_idx, col_idx) relative to top-left of face
    #        ┌────┐ (0,1)
    #        │ U  │
    #   ┌────┼────┼────┬────┐ (1,0) (1,1) (1,2) (1,3)
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │ (2,1)
    #        └────┘
    placements = {
        Face.U: (0, 1), Face.L: (1, 0), Face.F: (1, 1),
        Face.R: (1, 2), Face.B: (1, 3), Face.D: (2, 1)
    }
    face_enum_order = [Face.F, Face.R, Face.B, Face.L, Face.U, Face.D]

    for face_idx, face_data in enumerate(cube_list):
         face_enum = face_enum_order[face_idx]
         row_idx, col_idx = placements[face_enum]
         row_start = row_idx * h
         col_start = col_idx * face_w
         if channels is not None:
             cube_dice[row_start:row_start+h, col_start:col_start+face_w, :] = face_data
         else:
             cube_dice[row_start:row_start+h, col_start:col_start+face_w] = face_data

    return cube_dice


def cube_dice2list(cube_dice: NDArray[DType]) -> list[NDArray[DType]]:
    """Convert a 'dice' format cubemap (cross layout) to a list of faces [F,R,B,L,U,D]."""
    h_total = cube_dice.shape[0]
    w_total = cube_dice.shape[1]
    if h_total % 3 != 0:
        raise ValueError(f"Dice image height ({h_total}) must be a multiple of 3.")
    if w_total % 4 != 0:
        raise ValueError(f"Dice image width ({w_total}) must be a multiple of 4.")

    face_h = h_total // 3
    face_w = w_total // 4
    # Allow non-square
    # if face_h != face_w:
    #     print(f"Warning: cube_dice2list input face is not square ({face_h}x{face_w})")

    #        ┌────┐ U(0,1)
    #        │ U  │
    #   ┌────┼────┼────┬────┐ L(1,0) F(1,1) R(1,2) B(1,3)
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │ D(2,1)
    #        └────┘
    locations = {
        "U": (0, 1), "L": (1, 0), "F": (1, 1),
        "R": (1, 2), "B": (1, 3), "D": (2, 1)
    }
    output_order = ["F", "R", "B", "L", "U", "D"] # Standard F R B L U D list order
    out_list = []

    has_channels = cube_dice.ndim == 3 and cube_dice.shape[-1] > 1

    for face_key in output_order:
        row_idx, col_idx = locations[face_key]
        row_start = row_idx * face_h
        col_start = col_idx * face_w
        if has_channels:
            face_data = cube_dice[row_start:row_start+face_h, col_start:col_start+face_w, :]
        else: # 2D dice image
            face_data = cube_dice[row_start:row_start+face_h, col_start:col_start+face_w]
        out_list.append(face_data.copy()) # Return copies

    return out_list


def cube_dice2h(cube_dice: NDArray[DType]) -> NDArray[DType]:
    """Convert a 'dice' format cubemap to a horizontal strip."""
    return cube_list2h(cube_dice2list(cube_dice))


# --- Geometric Transformation Functions (from original utils.py) ---

def rotation_matrix(rad: float, ax: Union[int, str, NDArray, Sequence]) -> NDArray[np.float64]:
    """Creates a 3x3 rotation matrix for rotation around a given axis.

    Args:
        rad: Angle of rotation in radians.
        ax: Axis of rotation. Can be an integer (0 for X, 1 for Y, 2 for Z),
            a string ('x', 'y', 'z'), or a 3-element sequence/array representing the axis vector.

    Returns:
        A 3x3 numpy array representing the rotation matrix (float64).
    """
    axis_map_int = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0]}
    axis_map_str = {'x': [1.0, 0.0, 0.0], 'y': [0.0, 1.0, 0.0], 'z': [0.0, 0.0, 1.0]}

    if isinstance(ax, int):
        if ax not in axis_map_int:
            raise ValueError("Integer axis must be 0 (X), 1 (Y), or 2 (Z).")
        ax_vec = axis_map_int[ax]
    elif isinstance(ax, str):
        ax_lower = ax.lower()
        if ax_lower not in axis_map_str:
             raise ValueError("String axis must be 'x', 'y', or 'z'.")
        ax_vec = axis_map_str[ax_lower]
    else:
        ax_vec = ax # Assume it's a sequence/array

    ax_vec = np.asarray(ax_vec, dtype=float)
    if ax_vec.shape != (3,):
        raise ValueError(f"Rotation axis must be a 3-element vector; got shape {ax_vec.shape}")

    # Normalize axis vector
    norm = np.linalg.norm(ax_vec)
    if norm < np.finfo(float).eps:
        # Return identity if axis is zero vector or close to it
        return np.identity(3, dtype=np.float64)
    ax_vec = ax_vec / norm

    # Use scipy's Rotation class for robust calculation
    try:
        rot = Rotation.from_rotvec(rad * ax_vec)
        return rot.as_matrix().astype(np.float64)
    except Exception as e:
        raise ValueError(f"Failed to create rotation matrix: {e}") from e


# --- Equirectangular to Cubemap Conversion (from equirectangulartocubemap.py) ---

@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["horizon", "dice"] = "dice",
) -> NDArray[DType]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["list"] = "list",
) -> List[NDArray[DType]]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["dict"] = "dict",
) -> Dict[str, NDArray[DType]]: ...


def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: CubeFormat = "dice",
) -> Union[NDArray[DType], List[NDArray[DType]], Dict[str, NDArray[DType]]]:
    """Convert equirectangular image to cubemap using the above utilities.

    Parameters
    ----------
    e_img: ndarray
        Equirectangular image in shape of [H,W] (grayscale) or [H, W, C] (color).
    face_w: int
        Length of each face of the output cubemap (output faces will be face_w x face_w).
    mode: InterpolationMode
        Interpolation mode (e.g., "nearest", "bilinear", "bicubic").
    cube_format: CubeFormat
        Format to return cubemap in ("horizon", "list", "dict", "dice").

    Returns
    -------
    Union[NDArray, list[NDArray], dict[str, NDArray]]
        Cubemap in format specified by `cube_format`.
    """
    if e_img.ndim not in (2, 3):
        raise ValueError("e_img must have 2 or 3 dimensions (H,W or H,W,C).")

    h, w = e_img.shape[:2]
    order = mode_to_order(mode)

    # Get the sampler instance (uses cached coordinates if possible, though caching disabled now)
    # This sampler precomputes the sampling coordinates from equirectangular for a cubemap layout
    sampler = EquirecSampler.from_cubemap(face_w, h, w, order)

    # Apply the sampler to the image
    # The sampler's __call__ handles 2D/3D input and channels internally.
    cubemap_h_strip = sampler(e_img) # Output is (face_w, face_w*6) or (face_w, face_w*6, C)

    # Convert the resulting horizontal strip to the desired output format
    if cube_format == "horizon":
        return cubemap_h_strip
    elif cube_format == "list":
        return cube_h2list(cubemap_h_strip) # List of (face_w, face_w) or (face_w, face_w, C) arrays
    elif cube_format == "dict":
        return cube_h2dict(cubemap_h_strip) # Dict of face arrays
    elif cube_format == "dice":
        return cube_h2dice(cubemap_h_strip) # (face_w*3, face_w*4) or (..., C) array
    else:
        # Should be caught by Literal typing, but defensively check
        raise ValueError(f"Unknown cube_format: {cube_format}")


# --- Equirectangular to Perspective Conversion (from e2p.py) ---

def e2p(
    e_img: NDArray[DType],
    fov_deg: Union[float, int, Tuple[Union[float, int], Union[float, int]]],
    u_deg: float,
    v_deg: float,
    out_hw: Tuple[int, int],
    roll_deg: float = 0.0, # Changed name from in_rot_deg for clarity
    mode: InterpolationMode = "bilinear",
) -> NDArray[DType]:
    """Convert equirectangular image to perspective view.

    Parameters
    ----------
    e_img: ndarray
        Equirectangular image in shape of [H, W] (grayscale) or [H, W, C] (color).
    fov_deg: Union[float, int, Tuple[float, float]]
        Field of view in degrees. Can be a single value (applied to both horizontal
        and vertical) or a tuple (h_fov_deg, v_fov_deg).
    u_deg: float
        Horizontal viewing angle (azimuth) in degrees. Range [-180, 180].
        0 is center, positive is right, negative is left.
    v_deg: float
        Vertical viewing angle (elevation) in degrees. Range [-90, 90].
        0 is horizon, positive is up, negative is down.
    out_hw: Tuple[int, int]
        Size of the output perspective image (height, width).
    roll_deg: float, optional
        In-plane rotation (roll) in degrees. Default is 0. Positive is clockwise.
    mode: InterpolationMode, optional
        Interpolation mode (e.g., "nearest", "bilinear", "bicubic"). Default "bilinear".

    Returns
    -------
    np.ndarray
        Perspective image in shape [out_h, out_w] or [out_h, out_w, C].
    """
    if e_img.ndim not in (2, 3):
        raise ValueError("e_img must have 2 or 3 dimensions (H,W or H,W,C).")

    in_h, in_w = e_img.shape[:2]
    out_h, out_w = out_hw
    order = mode_to_order(mode)

    # Parse FOV and convert to radians
    if isinstance(fov_deg, (int, float, Real)):
        # Assume square pixel aspect ratio if only one FOV is given
        h_fov_rad = float(np.radians(fov_deg))
        v_fov_rad = h_fov_rad # Keep square FOV for single input
        # Alternative: Calculate v_fov based on aspect ratio of out_hw?
        # v_fov_rad = 2 * np.arctan(np.tan(h_fov_rad / 2) * (out_h / out_w))
    elif isinstance(fov_deg, tuple) and len(fov_deg) == 2:
        h_fov_rad = float(np.radians(fov_deg[0]))
        v_fov_rad = float(np.radians(fov_deg[1]))
    else:
        raise ValueError("fov_deg must be a single number or a tuple of two numbers (h_fov, v_fov).")

    # Convert angles to radians
    # User input: u_deg positive right, v_deg positive up, roll_deg positive clockwise
    # Internal coord system (xyzpers/rotation_matrix):
    #   u (azimuth) is rotation around Y axis (positive counter-clockwise looking from +Y)
    #   v (elevation) is rotation around X axis (positive counter-clockwise looking from +X)
    #   roll is rotation around Z axis (positive counter-clockwise looking from +Z)
    # Mapping:
    #   User right -> Negative Y rotation -> u_rad = -radians(u_deg)
    #   User up    -> Positive X rotation -> v_rad = radians(v_deg)
    #   User clockwise roll -> Negative Z rotation -> roll_rad = -radians(roll_deg)
    u_rad = -np.radians(u_deg)
    v_rad = np.radians(v_deg)
    roll_rad = -np.radians(roll_deg) # Adjusted based on xyzpers/rotation matrix conventions


    # Get the sampler instance (uses cached coordinates if possible, though caching disabled now)
    sampler = EquirecSampler.from_perspective(
        h_fov_rad, v_fov_rad, u_rad, v_rad, roll_rad,
        in_h, in_w, out_h, out_w, order
    )

    # Apply the sampler to the image
    # The sampler's __call__ handles 2D/3D input and channels internally.
    pers_img = sampler(e_img)

    return pers_img


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing combined projection utilities...")

    # Create a dummy equirectangular image (e.g., gradient)
    h_eq, w_eq = 512, 1024
    dummy_eq = np.zeros((h_eq, w_eq, 3), dtype=np.float32)
    gy, gx = np.mgrid[0:1:h_eq*1j, 0:1:w_eq*1j]
    dummy_eq[..., 0] = gx # Red channel = horizontal position
    dummy_eq[..., 1] = gy # Green channel = vertical position
    dummy_eq[..., 2] = np.sin(gx * 4 * np.pi) * 0.5 + 0.5 # Blue channel = vertical stripes

    face_w = 256
    print(f"\n--- Testing Equirectangular to Cubemap (e2c) ---")
    print(f"Input Equirectangular: {h_eq}x{w_eq}, Output Face Width: {face_w}")

    try:
        # Test e2c with different formats
        print("\nTesting e2c -> list (bilinear)")
        cube_list = e2c(dummy_eq, face_w=face_w, mode='bilinear', cube_format='list')
        print(f"  Output type: {type(cube_list)}, Number of faces: {len(cube_list)}")
        if cube_list: print(f"  Face shape: {cube_list[0].shape}, dtype: {cube_list[0].dtype}")

        print("\nTesting e2c -> dict (bicubic)")
        cube_dict = e2c(dummy_eq, face_w=face_w, mode='bicubic', cube_format='dict')
        print(f"  Output type: {type(cube_dict)}, Keys: {list(cube_dict.keys())}")
        if cube_dict: print(f"  Face shape: {cube_dict['F'].shape}, dtype: {cube_dict['F'].dtype}")

        print("\nTesting e2c -> horizon (nearest)")
        cube_horizon = e2c(dummy_eq, face_w=face_w, mode='nearest', cube_format='horizon')
        print(f"  Output type: {type(cube_horizon)}, Shape: {cube_horizon.shape}, dtype: {cube_horizon.dtype}")

        print("\nTesting e2c -> dice (bilinear)")
        cube_dice = e2c(dummy_eq, face_w=face_w, mode='bilinear', cube_format='dice')
        print(f"  Output type: {type(cube_dice)}, Shape: {cube_dice.shape}, dtype: {cube_dice.dtype}")

        # Test conversion back (if needed)
        print("\nTesting dice -> list")
        re_list = cube_dice2list(cube_dice)
        print(f"  Output type: {type(re_list)}, Faces: {len(re_list)}")
        if re_list: print(f"  Face shape: {re_list[0].shape}")

        print("\nTesting list -> horizon")
        re_h = cube_list2h(cube_list)
        print(f"  Output type: {type(re_h)}, Shape: {re_h.shape}")

        # Test grayscale
        print("\nTesting e2c (grayscale) -> list")
        dummy_eq_gray = dummy_eq.mean(axis=-1).astype(np.float32) # H,W
        cube_list_gray = e2c(dummy_eq_gray, face_w=face_w, mode='bilinear', cube_format='list')
        print(f"  Output type: {type(cube_list_gray)}, Number of faces: {len(cube_list_gray)}")
        if cube_list_gray: print(f"  Face shape: {cube_list_gray[0].shape}, dtype: {cube_list_gray[0].dtype}")


        # Test CubeFaceSampler (Cubemap -> Equirectangular) - requires scipy
        try:
            print("\n--- Testing Cubemap to Equirectangular (CubeFaceSampler) ---")
            # Use the list generated earlier
            c2e_sampler = CubeFaceSampler.from_equirec(face_w=face_w, h=h_eq, w=w_eq, order=1) # Bilinear
            # Need cube faces as (6, S, S, C) or (6, S, S)
            cube_faces_array = np.stack(cube_list, axis=0) # Shape (6, face_w, face_w, 3)
            reconstructed_eq = c2e_sampler(cube_faces_array)
            print(f"  Reconstructed equirectangular shape: {reconstructed_eq.shape}, dtype: {reconstructed_eq.dtype}")
            # You could save/visualize reconstructed_eq vs dummy_eq here
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 1); plt.imshow(dummy_eq); plt.title('Original Equirect')
            # plt.subplot(1, 2, 2); plt.imshow(reconstructed_eq); plt.title('Reconstructed Equirect')
            # plt.show()
        except NameError as ne:
             if 'map_coordinates' in str(ne):
                 print("  Skipping CubeFaceSampler test: Requires SciPy")
             else: raise ne
        except Exception as e_c2e:
             print(f"  Error during CubeFaceSampler test: {e_c2e}")


        # --- Test Equirectangular to Perspective (e2p) ---
        print("\n--- Testing Equirectangular to Perspective (e2p) ---")
        persp_h, persp_w = 480, 640
        fov = 90.0
        u_angle = 30.0 # Look 30 degrees right
        v_angle = -15.0 # Look 15 degrees down
        roll_angle = 10.0 # Roll 10 degrees clockwise

        print(f"Input Equirectangular: {h_eq}x{w_eq}")
        print(f"Output Perspective: {persp_h}x{persp_w}, FoV: {fov} deg")
        print(f"View Angles: u={u_angle} deg, v={v_angle} deg, roll={roll_angle} deg")

        print("\nTesting e2p (color, bilinear)")
        persp_img_color = e2p(
            dummy_eq, fov_deg=fov, u_deg=u_angle, v_deg=v_angle,
            out_hw=(persp_h, persp_w), roll_deg=roll_angle, mode='bilinear'
        )
        print(f"  Output shape: {persp_img_color.shape}, dtype: {persp_img_color.dtype}")

        print("\nTesting e2p (grayscale, nearest, tuple FOV)")
        persp_img_gray = e2p(
            dummy_eq_gray, fov_deg=(100, 80), u_deg=0, v_deg=0,
            out_hw=(persp_h, persp_w), roll_deg=0, mode='nearest'
        )
        print(f"  Output shape: {persp_img_gray.shape}, dtype: {persp_img_gray.dtype}")

        # Optional visualization
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1); plt.imshow(persp_img_color); plt.title(f'Perspective Color (u={u_angle}, v={v_angle}, roll={roll_angle})')
        # plt.subplot(1, 2, 2); plt.imshow(persp_img_gray, cmap='gray'); plt.title('Perspective Gray (u=0, v=0, FoV=(100,80))')
        # plt.show()


        print("\nUtility tests completed.")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()