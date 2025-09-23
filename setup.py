from setuptools import setup, find_packages

setup(
    name="visualife",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
    author="Jeewan Aidi",
    description="VisuaLife Engine: Custom Neural Network Library from scratch",
    python_requires='>=3.8',
)
