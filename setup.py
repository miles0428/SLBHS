from setuptools import find_packages, setup

install_requires = [
    "numpy>=1.24.0",
    "scikit-learn==1.3.2",
    "h5py>=3.10.0",
    "umap-learn==0.5.6",  # 0.5.8+ incompatible with scikit-learn 1.3.2
    "matplotlib>=3.7.0",
    "scipy",
]

setup(
    name="SLBHS",
    version="0.1.06",
    packages=["SLBHS", "SLBHS.data", "SLBHS.clustering", "SLBHS.viz"],
    package_dir={"": "."},
    install_requires=install_requires,
    author="Yu-Cheng Chung",
    description="Sign Language Basic Handshapes — hand pose clustering and UMAP visualization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/miles0428/SLBHS",
    license="SLBHS License (see LICENSE file)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "slbhs=SLBHS.run_visualization:main",
        ],
    },
)
