import os
import re
from setuptools import setup, find_packages

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "src", "aweSOM", "__init__.py"), encoding="utf-8") as f:
    version = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M).group(1)

# Read the requirements, skipping blank lines and comments
with open(os.path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
    # pip comment rule: "#" at line start or after whitespace (keeps URL #fragments)
    required = [re.sub(r"(^|\s)#.*$", "", line).strip() for line in f]
    required = [line for line in required if line]

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aweSOM",
    version=version,
    author="Trung Ha",
    author_email="thccrg@rit.edu",
    description="Accelerated Self-Organizing Maps and Statistically Combined Ensemble method for plasma intermittency detection",
    long_description=long_description,
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
    extras_require={"gpu": ["jax[cuda12]"], "cuda": ["cupy-cuda12x"]},
)
