# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


''' simple utilities for retrieving image transforms '''
import SimpleITK as sitk
import torch
import numpy as np
from typing import Optional, Tuple
from fireants.utils.globals import PERMITTED_ANTS_MAT_EXT, PERMITTED_ANTS_TXT_EXT
from fireants.utils.util import any_extension
import logging

logger = logging.getLogger(__name__)


def get_affine_transform_from_file(
    filename: str,
    dim: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Load an ANTs affine transform file and convert to PyTorch tensor.

    Reads a SimpleITK/ANTs compatible transform file (.mat or .txt) and converts
    it to a PyTorch tensor suitable for use with FireANTs registration classes.

    Args:
        filename: Path to the transform file (.mat or .txt format)
        dim: Number of spatial dimensions (2 or 3). Default is 3.
        device: Target device for the tensor. Default is CPU.
        dtype: Data type for the tensor. Default is torch.float32.

    Returns:
        torch.Tensor: Affine matrix of shape [1, dim, dim+1] ready for use
            with init_affine or init_rigid parameters in registration classes.

    Raises:
        ValueError: If the file extension is not supported or transform type
            is not an affine transform.

    Example:
        >>> init_affine = get_affine_transform_from_file("transform.mat", dim=3)
        >>> reg = GreedyRegistration(..., init_affine=init_affine)
    """
    MAT_TXT_FILES = PERMITTED_ANTS_MAT_EXT + PERMITTED_ANTS_TXT_EXT
    if not any_extension(filename, MAT_TXT_FILES):
        raise ValueError(f"File {filename} is not an ANTs transform file. "
                        f"Supported extensions: {MAT_TXT_FILES}")

    # Read the transform using SimpleITK
    transform = sitk.ReadTransform(filename)

    # Get transform type and validate
    transform_name = transform.GetName()
    logger.info(f"Loaded transform: {transform_name} from {filename}")

    # Handle composite transforms (take the first one)
    if isinstance(transform, sitk.CompositeTransform):
        if transform.GetNumberOfTransforms() == 0:
            raise ValueError(f"Composite transform in {filename} has no transforms")
        transform = transform.GetNthTransform(0)
        transform_name = transform.GetName()
        logger.info(f"Extracted first transform from composite: {transform_name}")

    # Extract parameters based on transform type
    if 'Affine' in transform_name or 'affine' in transform_name.lower():
        affine_tensor = _extract_affine_parameters(transform, dim, device, dtype)
    elif 'Euler' in transform_name or 'Rigid' in transform_name:
        # Euler/Rigid transforms can be converted to affine
        affine_tensor = _extract_rigid_as_affine(transform, dim, device, dtype)
    elif 'Translation' in transform_name:
        affine_tensor = _extract_translation_as_affine(transform, dim, device, dtype)
    else:
        # Try to get it as a generic linear transform
        try:
            affine_tensor = _extract_affine_parameters(transform, dim, device, dtype)
        except Exception as e:
            raise ValueError(f"Unsupported transform type: {transform_name}. "
                           f"Only affine-like transforms are supported. Error: {e}")

    return affine_tensor


def _extract_affine_parameters(
    transform: sitk.Transform,
    dim: int,
    device: Optional[torch.device],
    dtype: torch.dtype
) -> torch.Tensor:
    """Extract affine parameters from a SimpleITK transform."""
    # Get the matrix and translation components
    # For AffineTransform, GetMatrix returns D*D values (row-major)
    # GetTranslation returns D values
    matrix = np.array(transform.GetMatrix(), dtype=np.float64).reshape(dim, dim)
    translation = np.array(transform.GetTranslation(), dtype=np.float64)

    # Get fixed parameters (center of rotation) if available
    fixed_params = transform.GetFixedParameters()
    if len(fixed_params) == dim:
        center = np.array(fixed_params, dtype=np.float64)
    else:
        center = np.zeros(dim, dtype=np.float64)

    # ANTs/ITK convention: the transform maps points as:
    #   y = A * (x - center) + center + translation
    #   y = A*x + (center - A*center + translation)
    #   y = A*x + b  where b = center - A*center + translation
    #
    # We need to compose this into a single [dim, dim+1] matrix
    effective_translation = center - matrix @ center + translation

    # Build the [1, dim, dim+1] tensor
    affine = np.zeros((1, dim, dim + 1), dtype=np.float64)
    affine[0, :, :dim] = matrix
    affine[0, :, dim] = effective_translation

    # Convert to torch tensor
    affine_tensor = torch.tensor(affine, dtype=dtype)
    if device is not None:
        affine_tensor = affine_tensor.to(device)

    return affine_tensor


def _extract_rigid_as_affine(
    transform: sitk.Transform,
    dim: int,
    device: Optional[torch.device],
    dtype: torch.dtype
) -> torch.Tensor:
    """Extract rigid transform parameters and return as affine matrix."""
    # Rigid transforms have GetMatrix() and GetTranslation() like affine
    return _extract_affine_parameters(transform, dim, device, dtype)


def _extract_translation_as_affine(
    transform: sitk.Transform,
    dim: int,
    device: Optional[torch.device],
    dtype: torch.dtype
) -> torch.Tensor:
    """Extract translation-only transform as affine matrix."""
    translation = np.array(transform.GetOffset(), dtype=np.float64)

    # Build identity rotation with translation
    affine = np.zeros((1, dim, dim + 1), dtype=np.float64)
    affine[0, :, :dim] = np.eye(dim)
    affine[0, :, dim] = translation

    affine_tensor = torch.tensor(affine, dtype=dtype)
    if device is not None:
        affine_tensor = affine_tensor.to(device)

    return affine_tensor


def get_affine_transform_from_file_homogeneous(
    filename: str,
    dim: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Load transform and return as homogeneous matrix [1, dim+1, dim+1].

    Same as get_affine_transform_from_file but returns a full homogeneous
    matrix suitable for matrix multiplication chains.

    Args:
        filename: Path to the transform file
        dim: Number of spatial dimensions (2 or 3)
        device: Target device for the tensor
        dtype: Data type for the tensor

    Returns:
        torch.Tensor: Homogeneous matrix of shape [1, dim+1, dim+1]
    """
    affine = get_affine_transform_from_file(filename, dim, device, dtype)

    # Add homogeneous row [0, 0, ..., 0, 1]
    batch_size = affine.shape[0]
    row = torch.zeros((batch_size, 1, dim + 1), dtype=dtype)
    row[:, 0, -1] = 1.0
    if device is not None:
        row = row.to(device)

    homogeneous = torch.cat([affine, row], dim=1)
    return homogeneous