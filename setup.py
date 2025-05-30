from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11

setup(
  name = 'ring-attention-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.5.20',
  license='MIT',
  description = 'Ring Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/ring-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'distributed attention'
  ],
  install_requires=[
    'beartype',
    'einops>=0.8.0',
    'jaxtyping',
    'torch>=2.0',
  ],
  ext_modules=[
    CUDAExtension(
      name='ring_attention_pytorch._tree_attn_cuda_ext',
      sources=[
        'ring_attention_pytorch/csrc/tree_attn_cuda.cpp',
        'ring_attention_pytorch/csrc/tree_attn_cuda_kernel.cu',
        'ring_attention_pytorch/csrc/tree_attn_cuda_kernel_fused.cu',
      ],
      include_dirs=[pybind11.get_include()],
      extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3']
      }
    )
  ],
  cmdclass={
    'build_ext': BuildExtension
  },
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
