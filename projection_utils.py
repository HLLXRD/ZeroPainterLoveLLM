# Combined utilities for equirectangular/cubemap conversion and related geometry
# Based on the provided utils.py and equirectangulartocubemap.py snippets.

from collections.abc import Sequence
from enum import IntEnum
from functools import lru_cache
from typing import Any, Literal, Optional, TypeVar, Union, overload, Dict, List

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


# @lru_cache(_CACHE_SIZE)
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
    out.setflags(write=False)
    return out


# @lru_cache(_CACHE_SIZE)
def equirect_uvgrid(h: int, w: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi / 2, -np.pi / 2, num=h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    # Since we are using lru_cache, we want the return value to be immutable.
    uu.setflags(write=False)
    vv.setflags(write=False)
    return uu, vv  # pyright: ignore[reportReturnType]


# @lru_cache(_CACHE_SIZE)
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
    remainder = w - s.stop  # pyright: ignore[reportPossiblyUnboundVariable]
    tp[:h3, s.stop :][mask[:, :remainder]] = Face.UP  # pyright: ignore[reportPossiblyUnboundVariable]
    tp[-h3:, s.stop :][flip_mask[:, :remainder]] = Face.DOWN  # pyright: ignore[reportPossiblyUnboundVariable]

    # Since we are using lru_cache, we want the return value to be immutable.
    tp.setflags(write=False)

    return tp


def xyzpers(
    h_fov: float, v_fov: float, u: float, v: float, out_hw: tuple[int, int], in_rot: float
) -> NDArray[np.float32]:
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, Dim.X)
    Ry = rotation_matrix(u, Dim.Y)
    # Calculate the intrinsic rotation axis after Rx and Ry have been applied
    initial_z_axis = np.array([0, 0, 1.0])
    rotated_z_axis = initial_z_axis @ Ry.T @ Rx.T # Order matters for extrinsic -> intrinsic
    Ri = rotation_matrix(in_rot, rotated_z_axis)

    # Apply rotations: Intrinsic (Ri), then Extrinsic (Rx, Ry)
    # Equivalent to World -> Ry -> Rx -> Ri -> Camera
    # Or: apply Ri to points, then Rx, then Ry to points
    # out = out @ Ri.T @ Rx.T @ Ry.T # Apply inverse transforms to points
    out = out @ Ri.T # Apply intrinsic rotation first to coordinates
    out = out @ Rx.T # Then extrinsic X rotation
    out = out @ Ry.T # Then extrinsic Y rotation

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
    x = xyz[..., 0:1]  # Keep dimensions but avoid copy
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]
    u = np.arctan2(x, z)  # Angle in xz plane: 0 along +z, pi/2 along +x, pi along -z, -pi/2 along -x
    c = np.hypot(x, z)    # Distance in xz plane
    v = np.arctan2(y, c)  # Angle from xz plane towards y
    return u, v


def uv2unitxyz(uv: NDArray[DType]) -> NDArray[DType]:
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
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi  # Map x to u
    v = -((coor_y + 0.5) / h - 0.5) * np.pi  # Map y to v
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
                raise NotImplementedError

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
            print(f"Warning: EquirecSampler promoting input dtype {source_dtype} to float32 for interpolation.")
            img_proc = img.astype(np.float32)
        else:
            img_proc = img

        # Pad the image
        padded = self._pad(img_proc)

        # Determine if image has channels
        has_channels = padded.ndim == 3
        if not has_channels:
            # Temporarily add channel dim for consistent processing
             padded = padded[..., None]

        channels = []
        num_channels = padded.shape[2]

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
                    borderMode=cv2.BORDER_REPLICATE # Or another border mode if needed
                )
            else:
                # map_coordinates expects data, coords=(y,x,...), order
                # coords shape should be (ndim, ...) where ... is output shape
                sampled_channel = map_coordinates(
                    channel_data,
                    self._coords_for_scipy, # Already has shape (2, H_out, W_out)
                    order=self._order,
                    mode='wrap', # Use wrap for equirectangular horizontal continuity
                    prefilter=True if self._order > 1 else False # Prefilter for higher orders
                )
            channels.append(sampled_channel)

        # Stack channels back if needed
        if num_channels > 1:
            out = np.stack(channels, axis=-1)
        else:
            out = channels[0] # Single channel, shape is already (H_out, W_out)

        # Ensure output shape matches coordinates shape exactly (sometimes off by one due to rounding?)
        # This shouldn't be necessary if coordinates were generated correctly
        if out.shape[:2] != self._shape:
             print(f"Warning: Output shape {out.shape[:2]} differs slightly from coordinate shape {self._shape}. Resizing.")
             # This might indicate an issue elsewhere, but resize as a fallback
             if cv2:
                 out = cv2.resize(out, (self._shape[1], self._shape[0]), interpolation=self._order if self._use_cv2 else cv2.INTER_LINEAR)
             else: # Fallback numpy resize (less ideal)
                 # A proper resize isn't trivial here, maybe error instead?
                 print("Error: Cannot easily resize without OpenCV. Check coordinate generation.")
                 # Or attempt simple slicing/padding - risky
                 # out = out[:self._shape[0], :self._shape[1]] # Simplistic crop

        # Cast back to original dtype if needed
        if out.dtype != source_dtype:
            if np.issubdtype(source_dtype, np.integer):
                 out = np.clip(np.round(out), np.iinfo(source_dtype).min, np.iinfo(source_dtype).max)
            out = out.astype(source_dtype)

        return out

# Replace the existing _pad method in projection_utils.py EquirecSampler class

    def _pad(self, img: NDArray) -> NDArray:
        """Adds 1 pixel padding for interpolation, handling equirectangular wrapping.

        Pads manually: 'wrap' for horizontal (W), 'edge' for vertical (H).
        """
        h, w = img.shape[:2]
        num_dims = img.ndim
        channels = img.shape[2] if num_dims > 2 else None

        # 1. Pad Horizontally (Axis 1) with 'wrap'
        #    Padding width for axis 1 is (1, 1), others are (0, 0)
        pad_width_h = ((0, 0), (1, 1)) + ((0, 0),) * (num_dims - 2)
        padded_h = np.pad(img, pad_width_h, mode='wrap')

        # Check intermediate shape (should be H x W+2 x ...)
        # print(f"Shape after horizontal padding: {padded_h.shape}")

        # 2. Pad Vertically (Axis 0) with 'edge'
        #    Padding width for axis 0 is (1, 1), others are (0, 0)
        pad_width_v = ((1, 1), (0, 0)) + ((0, 0),) * (num_dims - 2)
        padded_final = np.pad(padded_h, pad_width_v, mode='edge')

        # Check final shape (should be H+2 x W+2 x ...)
        expected_shape_list = list(img.shape)
        expected_shape_list[0] += 2
        expected_shape_list[1] += 2
        expected_shape = tuple(expected_shape_list)
        if padded_final.shape != expected_shape:
             print(f"Warning: Padding resulted in unexpected shape {padded_final.shape}, expected {expected_shape}")

        return padded_final

    @classmethod
    # @lru_cache(_CACHE_SIZE)
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
        coor_x, coor_y = uv2coor(u, v, h, w) # Shapes (face_w, face_w * 6) each

        # Reshape coords to match output structure (e.g., horizontal strip)
        # Output shape will be (face_w, face_w * 6)
        return cls(coor_x.squeeze(-1), coor_y.squeeze(-1), order=order)

    @classmethod
    # @lru_cache(_CACHE_SIZE)
    def from_perspective(
        cls, h_fov: float, v_fov: float, u_deg: float, v_deg: float, roll_deg: float,
        in_h: int, in_w: int, out_h: int, out_w: int, order: int
    ):
        """Construct a EquirecSampler to sample equirectangular for a perspective view.

        Parameters
        ----------
        h_fov: float
            Horizontal field of view in degrees.
        v_fov: float
            Vertical field of view in degrees.
        u_deg: float
            Horizontal center viewing angle in degrees (azimuth). 0 is center, positive is right.
        v_deg: float
            Vertical center viewing angle in degrees (elevation). 0 is horizon, positive is up.
        roll_deg: float
            In-plane rotation in degrees.
        in_h: int
            Height of input equirec image.
        in_w: int
            Width of input equirec image.
        out_h: int
            Height of output perspective image.
        out_w: int
            Width of output perspective image.
        order: int
            The order of the spline interpolation.
        """
        # Convert angles to radians
        h_fov_rad = np.radians(h_fov)
        v_fov_rad = np.radians(v_fov)
        u_rad = np.radians(u_deg)
        v_rad = np.radians(v_deg)
        roll_rad = np.radians(roll_deg)

        # Get XYZ coords for the perspective grid
        xyz = xyzpers(h_fov_rad, v_fov_rad, u_rad, v_rad, (out_h, out_w), roll_rad) # Shape (out_h, out_w, 3)

        # Convert XYZ to UV angles (longitude, latitude)
        u, v = xyz2uv(xyz) # Shapes (out_h, out_w, 1) each

        # Convert UV angles to equirectangular pixel coordinates
        coor_x, coor_y = uv2coor(u, v, in_h, in_w) # Shapes (out_h, out_w) each

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
                raise NotImplementedError

            # For OpenCV, we need to map coordinates to a vertically stacked image.
            # The effective Y coordinate becomes face_index * (face_h + pad*2) + coor_y_pad
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
            print(f"Warning: CubeFaceSampler promoting input dtype {source_dtype} to float32 for interpolation.")
            cube_proc = cube_faces.astype(np.float32)
        else:
            cube_proc = cube_faces

        # Pad each face
        padded = self._pad(cube_proc) # Shape (6, S+2, S+2, C) or (6, S+2, S+2)

        # Determine if image has channels
        has_channels = padded.ndim == 4
        if not has_channels:
             padded = padded[..., None] # Add channel dim

        channels = []
        num_channels = padded.shape[3]

        for i in range(num_channels):
            channel_data = padded[..., i] # Shape (6, S+2, S+2)
            if self._use_cv2:
                 # Vertically stack the padded faces for cv2.remap
                 v_stack = channel_data.reshape(-1, self._face_w + 2) # Shape ((6*(S+2)), S+2)

                 # Ensure data type is compatible with cv2.remap
                 if v_stack.dtype not in (np.uint8, np.float32):
                     v_stack = v_stack.astype(np.float32)

                 sampled_channel = cv2.remap(
                    v_stack,
                    self._map_x,
                    self._map_y,
                    interpolation=self._order,
                    borderMode=cv2.BORDER_REPLICATE # Should sample within padded area
                )
            else:
                # map_coordinates takes data[face, y, x] and coords[dim, out_coords]
                sampled_channel = map_coordinates(
                    channel_data, # Shape (6, S+2, S+2)
                    self._coords_for_scipy, # Shape (3, H_out, W_out) corresponding to (face, y, x)
                    order=self._order,
                    mode='nearest', # Use nearest for face index, default for others
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

        return out # pyright: ignore[reportReturnType]


    def _pad(self, cube_faces: NDArray[DType]) -> NDArray[DType]:
        """Adds 1 pixel padding around each cube face, handling adjacent face connections."""
        S = self._face_w
        padded = np.pad(cube_faces, ((0, 0), (1, 1), (1, 1), (0,0)) if cube_faces.ndim == 4 else ((0,0), (1,1), (1,1)),
                         mode='empty') # Use empty initially

        # Define indices for easy access
        F, R, B, L, U, D = Face.FRONT, Face.RIGHT, Face.BACK, Face.LEFT, Face.UP, Face.DOWN
        TOP, BOTTOM, LEFT_EDGE, RIGHT_EDGE = 0, -1, 0, -1 # Indices within the padded (S+2, S+2) face

        # --- Fill Edges ---
        # Front face neighbours
        padded[F, TOP, 1:-1, ...]    = padded[U, BOTTOM-1, 1:-1, ...]    # Top edge from Up's bottom edge
        padded[F, BOTTOM, 1:-1, ...] = padded[D, TOP+1, 1:-1, ...]        # Bottom edge from Down's top edge
        padded[F, 1:-1, LEFT_EDGE, ...]  = padded[L, 1:-1, RIGHT_EDGE-1, ...] # Left edge from Left's right edge
        padded[F, 1:-1, RIGHT_EDGE, ...] = padded[R, 1:-1, LEFT_EDGE+1, ...]  # Right edge from Right's left edge

        # Right face neighbours
        padded[R, TOP, 1:-1, ...]    = np.flip(padded[U, 1:-1, RIGHT_EDGE-1, ...], axis=0) # Top edge from Up's right edge (flipped)
        padded[R, BOTTOM, 1:-1, ...] = padded[D, 1:-1, RIGHT_EDGE-1, ...]          # Bottom edge from Down's right edge
        padded[R, 1:-1, LEFT_EDGE, ...]  = padded[F, 1:-1, RIGHT_EDGE-1, ...]        # Left edge from Front's right edge
        padded[R, 1:-1, RIGHT_EDGE, ...] = padded[B, 1:-1, LEFT_EDGE+1, ...]         # Right edge from Back's left edge

        # Back face neighbours
        padded[B, TOP, 1:-1, ...]    = np.flip(padded[U, TOP+1, 1:-1, ...], axis=1) # Top edge from Up's top edge (flipped)
        padded[B, BOTTOM, 1:-1, ...] = np.flip(padded[D, BOTTOM-1, 1:-1, ...], axis=1) # Bottom edge from Down's bottom edge (flipped)
        padded[B, 1:-1, LEFT_EDGE, ...]  = padded[R, 1:-1, RIGHT_EDGE-1, ...]        # Left edge from Right's right edge
        padded[B, 1:-1, RIGHT_EDGE, ...] = padded[L, 1:-1, LEFT_EDGE+1, ...]         # Right edge from Left's left edge

        # Left face neighbours
        padded[L, TOP, 1:-1, ...]    = padded[U, 1:-1, LEFT_EDGE+1, ...]  # Top edge from Up's left edge
        padded[L, BOTTOM, 1:-1, ...] = np.flip(padded[D, 1:-1, LEFT_EDGE+1, ...], axis=0) # Bottom edge from Down's left edge (flipped)
        padded[L, 1:-1, LEFT_EDGE, ...]  = padded[B, 1:-1, RIGHT_EDGE-1, ...] # Left edge from Back's right edge
        padded[L, 1:-1, RIGHT_EDGE, ...] = padded[F, 1:-1, LEFT_EDGE+1, ...]  # Right edge from Front's left edge

        # Up face neighbours
        padded[U, TOP, 1:-1, ...]    = np.flip(padded[B, TOP+1, 1:-1, ...], axis=1) # Top edge from Back's top edge (flipped)
        padded[U, BOTTOM, 1:-1, ...] = padded[F, TOP+1, 1:-1, ...]        # Bottom edge from Front's top edge
        padded[U, 1:-1, LEFT_EDGE, ...]  = padded[L, TOP+1, 1:-1, ...]     # Left edge from Left's top edge
        padded[U, 1:-1, RIGHT_EDGE, ...] = np.flip(padded[R, TOP+1, 1:-1, ...], axis=0) # Right edge from Right's top edge (flipped)

        # Down face neighbours
        padded[D, TOP, 1:-1, ...]    = padded[F, BOTTOM-1, 1:-1, ...]     # Top edge from Front's bottom edge
        padded[D, BOTTOM, 1:-1, ...] = np.flip(padded[B, BOTTOM-1, 1:-1, ...], axis=1) # Bottom edge from Back's bottom edge (flipped)
        padded[D, 1:-1, LEFT_EDGE, ...]  = np.flip(padded[L, BOTTOM-1, 1:-1, ...], axis=0) # Left edge from Left's bottom edge (flipped)
        padded[D, 1:-1, RIGHT_EDGE, ...] = padded[R, BOTTOM-1, 1:-1, ...]      # Right edge from Right's bottom edge

        # --- Fill Corners ---
        # Corners generally need data from 2 other faces' edges.
        # Example: Front top-left corner needs Up's bottom-left and Left's top-right.
        # For simplicity here, we can often just replicate the adjacent edge pixel after edges are filled.
        padded[:, TOP, LEFT_EDGE, ...] = padded[:, TOP+1, LEFT_EDGE+1, ...]       # Top-left corner
        padded[:, TOP, RIGHT_EDGE, ...] = padded[:, TOP+1, RIGHT_EDGE-1, ...]     # Top-right corner
        padded[:, BOTTOM, LEFT_EDGE, ...] = padded[:, BOTTOM-1, LEFT_EDGE+1, ...]   # Bottom-left corner
        padded[:, BOTTOM, RIGHT_EDGE, ...] = padded[:, BOTTOM-1, RIGHT_EDGE-1, ...] # Bottom-right corner

        # Re-fill edges based on filled corners (to get the edge ends correct)
        padded[:, TOP, 1:-1, ...] = padded[:, TOP+1, 1:-1, ...]
        padded[:, BOTTOM, 1:-1, ...] = padded[:, BOTTOM-1, 1:-1, ...]
        padded[:, 1:-1, LEFT_EDGE, ...] = padded[:, 1:-1, LEFT_EDGE+1, ...]
        padded[:, 1:-1, RIGHT_EDGE, ...] = padded[:, 1:-1, RIGHT_EDGE-1, ...]

        return padded

    @classmethod
    # @lru_cache(_CACHE_SIZE)
    def from_equirec(cls, face_w: int, h: int, w: int, order: int):
        """Construct a CubeFaceSampler to sample cube faces for an equirectangular output.

        Parameters
        ----------
        face_w: int
            Length of each face of the input cubemap.
        h: int
            Output equirectangular image height.
        w: int
            Output equirectangular image width.
        order: int
            The order of the spline interpolation.
        """
        # Get UV grid for the output equirectangular image
        u, v = equirect_uvgrid(h, w) # u [-pi,pi], v [pi/2,-pi/2]

        # Get face id (0-5) for each pixel in the output equirectangular image
        tp = equirect_facetype(h, w) # Shape (h, w)

        # Calculate the coordinates (x, y) within the *source* cube face
        # corresponding to each (u, v) angle.
        # This is the inverse of the logic in EquirecSampler.from_cubemap

        coor_x = np.empty((h, w), dtype=np.float32)
        coor_y = np.empty((h, w), dtype=np.float32)
        face_w_half = face_w / 2.0

        # Use vectorized operations based on face type (tp)
        # The formulas project the (u,v) direction onto the cube faces.

        # Front face (tp == 0)
        mask = (tp == Face.FRONT)
        if np.any(mask):
            tan_u = np.tan(u[mask])
            tan_v = np.tan(v[mask])
            cos_u = np.cos(u[mask]) # Needed? Check derivation
            # Projected point on face: (x/z, y/z) scaled by face_w_half, relative to face center
            # z = 0.5 on front face. Assume origin at cube center.
            # xyz = (tan(u)*z, tan(v)*sqrt(x^2+z^2), z) = (tan(u)*0.5, tan(v)*sqrt(tan(u)^2*0.25 + 0.25), 0.5)
            # We need face coords: X = x + 0.5, Y = -y + 0.5 (scaled by face_w)
            # Face X coord ~ tan(u)
            # Face Y coord ~ -tan(v) / cos(u)
            coor_x[mask] = face_w_half * tan_u + face_w_half
            coor_y[mask] = face_w_half * (-tan_v / np.cos(u[mask])) + face_w_half

        # Right face (tp == 1)
        mask = (tp == Face.RIGHT)
        if np.any(mask):
             # Rotation: u' = u - pi/2. Project onto plane x=0.5. z = 0.5*tan(u'), y = 0.5*tan(v)/cos(u')
             # Face coords: X = -z + 0.5, Y = -y + 0.5 (relative to face center)
             u_prime = u[mask] - np.pi / 2
             tan_u_prime = np.tan(u_prime)
             tan_v = np.tan(v[mask])
             coor_x[mask] = face_w_half * (-tan_u_prime) + face_w_half
             coor_y[mask] = face_w_half * (-tan_v / np.cos(u_prime)) + face_w_half

        # Back face (tp == 2)
        mask = (tp == Face.BACK)
        if np.any(mask):
            # Rotation: u' = u - pi. Project onto plane z=-0.5. x = -0.5*tan(u'), y = -0.5*tan(v)/cos(u')
            # Face coords: X = -x + 0.5, Y = -y + 0.5
            u_prime = u[mask] - np.pi
            tan_u_prime = np.tan(u_prime)
            tan_v = np.tan(v[mask])
            coor_x[mask] = face_w_half * (tan_u_prime) + face_w_half # Note sign change for x
            coor_y[mask] = face_w_half * (-tan_v / np.cos(u_prime)) + face_w_half

        # Left face (tp == 3)
        mask = (tp == Face.LEFT)
        if np.any(mask):
            # Rotation: u' = u + pi/2. Project onto plane x=-0.5. z = 0.5*tan(u'), y = -0.5*tan(v)/cos(u')
            # Face coords: X = z + 0.5, Y = -y + 0.5
            u_prime = u[mask] + np.pi / 2
            tan_u_prime = np.tan(u_prime)
            tan_v = np.tan(v[mask])
            coor_x[mask] = face_w_half * (tan_u_prime) + face_w_half
            coor_y[mask] = face_w_half * (-tan_v / np.cos(u_prime)) + face_w_half

        # Up face (tp == 4)
        mask = (tp == Face.UP)
        if np.any(mask):
            # Project onto plane y=0.5. Use v' = pi/2 - v. dist = 0.5 * tan(v')
            # x = dist * sin(u), z = dist * cos(u)
            # Face coords: X = x + 0.5, Y = z + 0.5
            v_prime = np.pi/2 - v[mask]
            dist = face_w_half * np.tan(v_prime)
            coor_x[mask] = dist * np.sin(u[mask]) + face_w_half
            coor_y[mask] = dist * np.cos(u[mask]) + face_w_half # Y map corresponds to Z coord

        # Down face (tp == 5)
        mask = (tp == Face.DOWN)
        if np.any(mask):
            # Project onto plane y=-0.5. Use v' = pi/2 + v = pi/2 - abs(v)
            # dist = 0.5 * tan(v')
            # x = dist * sin(u), z = dist * cos(u)
            # Face coords: X = x + 0.5, Y = -z + 0.5
            v_prime = np.pi/2 + v[mask] # v is negative here
            dist = face_w_half * np.tan(v_prime)
            coor_x[mask] = dist * np.sin(u[mask]) + face_w_half
            coor_y[mask] = dist * (-np.cos(u[mask])) + face_w_half # Y map is -Z

        # Clip coordinates to be within the face bounds [0, face_w]
        # Note: Using pad (1) later, so valid range is effectively [-1, face_w] for sampling
        # Clipping to [0, face_w] ensures we don't sample too far into padding from invalid inputs.
        np.clip(coor_x, 0, face_w, out=coor_x)
        np.clip(coor_y, 0, face_w, out=coor_y)

        return cls(tp, coor_x, coor_y, order, face_w)


# --- Cubemap Format Conversion Functions (from original utils.py) ---

def cube_h2list(cube_h: NDArray[DType]) -> list[NDArray[DType]]:
    """Split a horizontal cubemap strip image into a list of 6 faces."""
    h = cube_h.shape[0]
    w_total = cube_h.shape[1]
    if w_total % 6 != 0:
        raise ValueError(f"Cubemap width ({w_total}) must be divisible by 6.")
    face_w = w_total // 6
    if h != face_w:
         # Allow non-square faces for flexibility, but maybe warn?
         # print(f"Warning: cube_h2list input height {h} != width/6 {face_w}")
         pass # Allow non-square faces if needed

    num_dims = cube_h.ndim
    if num_dims == 2: # Grayscale
        return [cube_h[:, slice_chunk(i, face_w)] for i in range(6)]
    elif num_dims == 3: # Color or Batched Grayscale? Assume Color HWC
        return [cube_h[:, slice_chunk(i, face_w), :] for i in range(6)]
    elif num_dims == 4: # Batched Color BHWC?
        return [cube_h[:, :, slice_chunk(i, face_w), :] for i in range(6)] # Adapt if B is first dim
    else:
        raise ValueError(f"Unsupported number of dimensions: {num_dims}")


def cube_list2h(cube_list: list[NDArray[DType]]) -> NDArray[DType]:
    """Concatenate a list of 6 face images side-by-side into a horizontal strip."""
    if len(cube_list) != 6:
        raise ValueError(f"6 elements must be provided to construct a cube; got {len(cube_list)}.")

    # Basic shape/dtype checks
    first_face = cube_list[0]
    first_shape = first_face.shape
    first_dtype = first_face.dtype
    num_dims = first_face.ndim

    for i, face in enumerate(cube_list[1:], 1):
        if face.shape != first_shape:
            raise ValueError(
                f"Face {i}'s shape {face.shape} doesn't match the first face's shape {first_shape}."
            )
        if face.dtype != first_dtype:
            raise ValueError(
                f"Face {i}'s dtype {face.dtype} doesn't match the first face's dtype {first_dtype}."
            )

    # Determine concatenation axis based on dimensions
    if num_dims == 2: # HW
        axis = 1
    elif num_dims == 3: # HWC or CHW? Assume HWC
        axis = 1
    elif num_dims == 4: # BHWC or BCHW? Assume BHWC
        axis = 2 # Concat along W dimension
        # If BCHW, axis would be 3
    else:
        raise ValueError(f"Unsupported number of dimensions: {num_dims}")

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
    if h != face_w:
        # print(f"Warning: cube_h2dice input height {h} != width/6 {face_w}")
        pass # Allow non-square

    cube_list = cube_h2list(cube_h)
    F, R, B, L, U, D = cube_list # Order is assumed F R B L U D

    out_h = h * 3
    out_w = face_w * 4
    channels = cube_h.shape[2] if cube_h.ndim > 2 else None

    if channels is not None:
        cube_dice = np.zeros((out_h, out_w, channels), dtype=cube_h.dtype)
    else:
        cube_dice = np.zeros((out_h, out_w), dtype=cube_h.dtype)

    # Define slices for dice layout (row_idx, col_idx) relative to top-left of face
    #        ┌────┐ (1,0)
    #        │ U  │
    #   ┌────┼────┼────┬────┐ (0,1) (1,1) (2,1) (3,1)
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │ (1,2)
    #        └────┘
    placements = {
        U: (0, 1), # Row 0, Col 1 (relative to face size)
        L: (1, 0), # Row 1, Col 0
        F: (1, 1), # Row 1, Col 1
        R: (1, 2), # Row 1, Col 2
        B: (1, 3), # Row 1, Col 3
        D: (2, 1)  # Row 2, Col 1
    }

    for face_data, (row_idx, col_idx) in zip(cube_list, [placements[k] for k in [U, L, F, R, B, D]]):
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
    if face_h != face_w:
        # print(f"Warning: cube_dice2list input face is not square ({face_h}x{face_w})")
        pass # Allow non-square

    # Define slices for dice layout (row_idx, col_idx) relative to top-left of face
    locations = {
        "U": (0, 1), "L": (1, 0), "F": (1, 1),
        "R": (1, 2), "B": (1, 3), "D": (2, 1)
    }
    output_order = ["F", "R", "B", "L", "U", "D"] # Standard output order
    out_list = []

    has_channels = cube_dice.ndim > 2

    for face_key in output_order:
        row_idx, col_idx = locations[face_key]
        row_start = row_idx * face_h
        col_start = col_idx * face_w
        if has_channels:
            face_data = cube_dice[row_start:row_start+face_h, col_start:col_start+face_w, :]
        else:
            face_data = cube_dice[row_start:row_start+face_h, col_start:col_start+face_w]
        out_list.append(face_data)

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
        A 3x3 numpy array representing the rotation matrix.
    """
    if isinstance(ax, int):
        if ax == 0: ax = [1.0, 0.0, 0.0]
        elif ax == 1: ax = [0.0, 1.0, 0.0]
        elif ax == 2: ax = [0.0, 0.0, 1.0]
        else: raise ValueError("Integer axis must be 0, 1, or 2.")
    elif isinstance(ax, str):
        ax = ax.lower()
        if ax == 'x': ax = [1.0, 0.0, 0.0]
        elif ax == 'y': ax = [0.0, 1.0, 0.0]
        elif ax == 'z': ax = [0.0, 0.0, 1.0]
        else: raise ValueError("String axis must be 'x', 'y', or 'z'.")

    ax = np.asarray(ax, dtype=float)
    if ax.shape != (3,):
        raise ValueError(f"ax must be shape (3,); got {ax.shape}")

    # Normalize axis vector
    norm = np.linalg.norm(ax)
    if norm == 0:
        # Return identity if axis is zero vector
        return np.identity(3)
    ax = ax / norm

    # Use scipy's Rotation class
    rot = Rotation.from_rotvec(rad * ax)
    return rot.as_matrix()


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
        Equirectangular image in shape of [H,W], [H, W, C], or potentially [B, H, W, C].
        Batch dimension (B) is experimental here.
    face_w: int
        Length of each face of the output cubemap.
    mode: InterpolationMode
        Interpolation mode (e.g., "nearest", "bilinear", "bicubic").
    cube_format: CubeFormat
        Format to return cubemap in ("horizon", "list", "dict", "dice").

    Returns
    -------
    Union[NDArray, list[NDArray], dict[str, NDArray]]
        Cubemap in format specified by `cube_format`.
    """
    if e_img.ndim not in (2, 3): # Basic check, Sampler handles more details
        raise ValueError("e_img must have 2 or 3 dimensions (H,W or H,W,C).")

    # Store original shape details
    input_shape = e_img.shape
    is_2d = e_img.ndim == 2

    # Temporarily add channel dim if 2D for consistent processing
    if is_2d:
        e_img_proc = e_img[..., None]
    else:
        e_img_proc = e_img

    h, w = e_img_proc.shape[:2]
    order = mode_to_order(mode)

    # Get the sampler instance (cached)
    # This sampler precomputes the sampling coordinates from equirectangular for a cubemap layout
    sampler = EquirecSampler.from_cubemap(face_w, h, w, order)

    # Apply the sampler to the image(s)
    # The sampler expects HWC or HW input.
    # Sampler's __call__ handles multiple channels internally.
    cubemap_h_strip = sampler(e_img_proc) # Output is HWC (face_w, face_w*6, C)

    # If the input was originally 2D, remove the channel dimension we added
    # if is_2d:
    #     cubemap_h_strip = cubemap_h_strip[..., 0] # Result is HW (face_w, face_w*6)

    # Convert the horizontal strip to the desired output format
    if cube_format == "horizon":
        # Already in horizontal strip format (or was before squeeze)
        return cubemap_h_strip # Type matches squeezed/unsqueezed state
    elif cube_format == "list":
        return cube_h2list(cubemap_h_strip) # List of HW or HWC arrays
    elif cube_format == "dict":
        return cube_h2dict(cubemap_h_strip) # Dict of HW or HWC arrays
    elif cube_format == "dice":
        return cube_h2dice(cubemap_h_strip) # HW or HWC array in dice layout
    else:
        # Should be caught by Literal typing, but defensively check
        raise ValueError(f"Unknown cube_format: {cube_format}")

# Example usage (optional, can be commented out)
if __name__ == '__main__':
    print("Testing combined projection utilities...")

    # Create a dummy equirectangular image (e.g., gradient)
    h_eq, w_eq = 512, 1024
    dummy_eq = np.zeros((h_eq, w_eq, 3), dtype=np.float32)
    gy, gx = np.mgrid[0:1:h_eq*1j, 0:1:w_eq*1j]
    dummy_eq[..., 0] = gx
    dummy_eq[..., 1] = gy
    dummy_eq[..., 2] = (gx + gy) / 2

    face_w = 256
    print(f"Converting dummy equirectangular ({h_eq}x{w_eq}) to cubemap (face_w={face_w})...")

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


        # Test CubeFaceSampler (Cubemap -> Equirectangular)
        print("\nTesting CubeFaceSampler (from list back to equirectangular)")
        # Use the list generated earlier
        c2e_sampler = CubeFaceSampler.from_equirec(face_w=face_w, h=h_eq, w=w_eq, order=1) # Bilinear
        # Need cube faces as (6, S, S, C) or (6, S, S)
        cube_faces_array = np.stack(cube_list, axis=0) # Shape (6, face_w, face_w, 3)
        reconstructed_eq = c2e_sampler(cube_faces_array)
        print(f"  Reconstructed equirectangular shape: {reconstructed_eq.shape}, dtype: {reconstructed_eq.dtype}")
        # You could save/visualize reconstructed_eq vs dummy_eq here

        print("\nUtility tests completed.")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()