from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "scikit-learn>=1.3.0",
    "h5py",
    "umap-learn",
    "matplotlib",
]

setup(
    name="SLBHS",
    version="0.1.01",
    packages=find_packages(where=["SLBHS"], include=["SLBHS", "SLBHS.*"]),
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
