from setuptools import setup
import re
from pathlib import Path


def get_version() -> str:
    with open(Path(__file__).parent / "msig" / "__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.split(r"['\"]", line)[1]


setup(
    name="msig",
    version=get_version(),
    description="Statistical Significance Criteria for multivariate Time Series Motifs",
    author="Miguel G. Silva",
    author_email="mmsilva@ciencias.ulisboa.pt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    keywords="motifs",
    packages=[
        "msig"
    ],
    install_requires=[
        "numpy",
        "scipy"
    ],
    url="https://github.com/MiguelGarcaoSilva/msig",
)