from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, load, include_paths
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
