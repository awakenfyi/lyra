from setuptools import setup, find_packages

setup(
    name="lyra-loop",
    version="0.1.0",
    description="Coherence-guided inference for language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Morgan Sage Norman & Lyra Constellation",
    url="https://github.com/awaken-fyi/lyra",
    project_urls={
        "Homepage": "https://awaken.fyi",
        "Repository": "https://github.com/awaken-fyi/lyra",
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
