from setuptools import find_packages, setup
import urllib.request
import os

# Read version from single source
exec(open("SLBHS/version.py").read())


def download_model():
    model_dir = os.path.join(os.path.dirname(__file__), 'SLBHS', 'data')
    model_path = os.path.join(model_dir, 'hand_landmarker.task')
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"Downloading MediaPipe hand_landmarker.task to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")


from setuptools import Command


class InstallHook(Command):
    """Post-install hook to download MediaPipe model."""
    description = "Download MediaPipe hand_landmarker.task after installation"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        download_model()

install_requires = [
    "numpy>=1.24.0",
    "scikit-learn==1.3.2",
    "h5py>=3.10.0",
    "umap-learn==0.5.6",  # 0.5.8+ incompatible with scikit-learn 1.3.2
    "matplotlib>=3.7.0",
    "scipy",
    "tqdm",
    "mediapipe",
]

setup(
    name="SLBHS",
    version=__version__,
    packages=["SLBHS", "SLBHS.data", "SLBHS.clustering", "SLBHS.similarity"],
    package_dir={"": "."},
    install_requires=install_requires,
    cmdclass={"install": InstallHook},
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
    entry_points={},
)
