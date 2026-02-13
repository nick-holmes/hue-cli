#!/usr/bin/env python3
"""
Setup script for HueCLI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="huecli",
    version="1.0.0",
    description="Generate multi-color 3D print STLs from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HueCLI Contributors",
    url="https://github.com/yourusername/huecli",  # Update with actual repo
    license="MIT",
    py_modules=["huecli", "generator"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "trimesh>=4.0.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "huecli=huecli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering",
    ],
    keywords="3d-printing stl multicolor filament heightmap mesh",
    include_package_data=True,
    zip_safe=False,
)
