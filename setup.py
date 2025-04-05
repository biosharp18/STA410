from setuptools import setup, find_packages

setup(
    name="genomic_optimization",
    version="0.1",
    author="Rory Gao",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pyfaidx",
        "tqdm",
        "ipykernel",
        "tangermeme",
        "bpnet-lite",
    ],
    description="A package for genomic optimization using gradient based methods",
    python_requires='>=3.9',
)