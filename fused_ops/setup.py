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


from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch
import os

include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')

class BuildExtensionWithFlags(cpp_extension.BuildExtension):
    def build_extensions(self):
        if getattr(self, 'debug', False):
            common_cxx_flags = ['-O0', '-g']
            common_nvcc_flags = ['-O0', '-G', '-g', '-lineinfo']
        else:
            common_cxx_flags = ['-O3']
            common_nvcc_flags = ['-O3']

        for ext in self.extensions:
            ext.extra_compile_args = {
                'cxx': common_cxx_flags,
                'nvcc': common_nvcc_flags,
            }

        super().build_extensions()

setup(
    name='fireants_fused_ops',
    version='1.0.0',
    description='Fused CUDA operations for FireANTs',
    author='Rohit Jena',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='fireants_fused_ops',
            sources=[
                'src/src.cpp',
                'src/CrossCorrelation.cu',  
                'src/FusedGridSampler.cu',
                'src/FusedGridComposer.cu',
                'src/FusedGenerateGrid.cu',
                'src/AdamUtils.cu',
                'src/GaussianBlurFFT.cu',
                'src/MutualInformation.cu'
            ],
            include_dirs=[include_dir] + torch.utils.cpp_extension.include_paths(),
            library_dirs=torch.utils.cpp_extension.library_paths(),
            extra_compile_args={}
        )
    ],
    cmdclass={'build_ext': BuildExtensionWithFlags},
    install_requires=['torch>=2.3.0'],
)
