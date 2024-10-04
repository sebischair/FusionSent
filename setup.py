from setuptools import setup, find_packages

# Read the contents of README file for 'long_description'.
with open("README.md", "r") as fh:
    _long_description = fh.read()

# Read the contents of requirements.txt for 'install_requires'.
with open('requirements.txt') as f:
    _requirements = f.read().splitlines()

setup(
    name="fusionsent",
    version="0.0.5", 
    author="Tim Schopf, Alexander Blatzheim",
    author_email="tim.schopf@tum.de, alexander.blatzheim@tum.de",
    description="FusionSent: A Fusion-Based Multi-Task Sentence Embedding Model",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NLP-Knowledge-Graph/few-shot-taxonomy-classification",
    packages=find_packages(),
    install_requires=_requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    license="Apache-2.0",
    include_package_data=True
)