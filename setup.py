from setuptools import setup
import re
from pathlib import Path


this_directory = Path(__file__).parent

def get_version() -> str:
    with open(this_directory / "msig" / "__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.split(r"['\"]", line)[1]

with open(this_directory / "README.rst") as f:
    long_description = f.read()

setup(
    name="msig",
    version=get_version(),
    description="Statistical Significance Criteria for multivariate Time Series Motifs",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author="Miguel G. Silva",
    author_email="mmsilva@ciencias.ulisboa.pt",
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3"
    ],
    keywords="time series motifs",
    packages=[
        "msig"
    ],
    install_requires=[
        "numpy",
        "scipy"
    ],
    url="https://github.com/MiguelGarcaoSilva/msig",
)