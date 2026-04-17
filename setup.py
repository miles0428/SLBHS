import os
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "scikit-learn>=1.3.0",
    "h5py",
    "umap-learn",
    "matplotlib",
]

if os.environ.get("TWSLT_ENABLE_CUDA", "0").lower() in ("1", "true", "yes"):
    install_requires.append("cupy")

setup(
    name="TWSLT",
    version="0.1.00",
    packages=find_packages(),
    install_requires=install_requires,
    author="Yu-Cheng Chung",
    description="Sign Language Basic Handshapes — hand pose clustering and UMAP visualization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/miles0428/SLBHS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
