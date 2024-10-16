from setuptools import setup, find_packages

# Read the contents of README file for 'long_description'.
with open("README.md", "r") as fh:
    _long_description = fh.read()

setup(
    name="fusionsent",
    version="0.0.8",
    author="Tim Schopf, Alexander Blatzheim",
    author_email="tim.schopf@tum.de, alexander.blatzheim@tum.de",
    description="FusionSent: A Fusion-Based Multi-Task Sentence Embedding Model",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sebischair/FusionSent",
    packages=find_packages(),
    install_requires= [
        "accelerate==0.32.1",
        "aiohappyeyeballs==2.4.3",
        "aiohttp==3.9.5",
        "aiosignal==1.3.1",
        "async-timeout==4.0.3",
        "attrs==23.2.0",
        "certifi==2024.8.30",
        "charset-normalizer==3.3.2",
        "datasets==2.20.0",
        "dill==0.3.5.1",
        "evaluate==0.4.2",
        "filelock==3.15.4",
        "frozenlist==1.4.1",
        "fsspec==2023.10.0",
        "huggingface-hub==0.21.2",
        "idna==3.10",
        "Jinja2==3.1.4",
        "joblib==1.4.2",
        "MarkupSafe==2.1.5",
        "mpmath==1.3.0",
        "multidict==6.0.5",
        "multiprocess==0.70.13",
        "networkx==3.3",
        "numpy==1.23.5",
        "packaging==24.1",
        "pandas==2.2.2",
        "pillow==10.4.0",
        "psutil==6.0.0",
        "pyarrow==15.0.0",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.2",
        "PyYAML==6.0.1",
        "regex==2024.5.15",
        "requests==2.32.3",
        "safetensors==0.4.3",
        "scikit-learn==1.5.1",
        "scipy==1.10.1",
        "sentence-transformers==3.0.1",
        "setfit==1.0.3",
        "six==1.16.0",
        "sympy==1.13.3",
        "threadpoolctl==3.5.0",
        "tokenizers==0.19.1",
        "torch==2.3.1",
        "tqdm==4.66.4",
        "transformers==4.40.0",
        "typing_extensions==4.12.2",
        "tzdata==2024.1",
        "urllib3==2.2.2",
        "xxhash==3.4.1",
        "yarl==1.9.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    license="Apache-2.0",
    include_package_data=True
)