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
from fireants.utils.globals import PERMITTED_ANTS_TRANSFORM_EXT
from fireants.utils.util import any_extension

def _extract_affine_from_sitk(transform: sitk.Transform, dim: int) -> torch.Tensor:
    if transform.GetDimension() != dim:
        raise ValueError(f"Transform dimension {transform.GetDimension()} does not match expected {dim}")
    matrix = torch.tensor(transform.GetMatrix(), dtype=torch.float32).reshape(dim, dim)
    translation = torch.tensor(transform.GetTranslation(), dtype=torch.float32).reshape(dim, 1)
    affine = torch.cat([matrix, translation], dim=1).unsqueeze(0)
    return affine


def get_affine_transform_from_file(filename: str, dim: int) -> torch.Tensor:
    if not any_extension(filename, PERMITTED_ANTS_TRANSFORM_EXT):
        raise ValueError(f"File {filename} is not an ANTS transform file")
    transform = sitk.ReadTransform(filename)
    if transform.GetTransformType() != "CompositeTransform":
        return _extract_affine_from_sitk(transform, dim)
    if transform.GetNumberOfTransforms() == 0:
        raise ValueError(f"Composite transform in {filename} is empty")
    inner_transform = transform.GetNthTransform(0)
    return _extract_affine_from_sitk(inner_transform, dim)
