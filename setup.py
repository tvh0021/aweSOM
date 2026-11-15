import os
import re
from setuptools import setup, find_packages

with open(
    os.path.join(os.path.dirname(__file__), "src", "aweSOM", "__init__.py"),
    encoding="utf-8",
) as f:
    version = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M).group(1)

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="aweSOM",
    version=version,
    author="Trung Ha",
    author_email="thccrg@rit.edu",
    description="Accelerated Self-Organizing Maps and Statistically Combined Ensemble method for plasma intermittency detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tvh0021/aweSOM.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=required,
    extras_require={"cuda": ["cupy-cuda12x"]},
    include_package_data=True,
)
