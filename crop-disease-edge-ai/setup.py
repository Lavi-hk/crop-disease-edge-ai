from setuptools import setup, find_packages

setup(
    name="crop-disease-edge-ai",
    version="1.0.0",
    description="Real-time crop disease detection for edge devices",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
    ],
)
