"""Package GRID-Planner while preserving its historical import namespace."""
import os
from pathlib import Path

import setuptools

# Allow installation from outside the checkout.
script_folder = Path(__file__).resolve().parent
os.chdir(script_folder)

requirements = [
    line.strip()
    for line in (script_folder / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setuptools.setup(
    name="grid-planner",
    version="1.0.0",
    description="GRID-Planner: Graph-Structured Relational Reward Shaping for Diffusion Planning",
    long_description=(script_folder / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Zhao Yang, Zezhong Qian, Yinan Shi, Longjun Liu, Yawei Jueluo",
    url="https://github.com/yangzhaojason/GRID-Planner",
    packages=setuptools.find_namespace_packages(
        include=["flow_planner", "flow_planner.*"],
        exclude=["*.__pycache__", "*.__pycache__.*"],
    ),
    package_dir={"": "."},
    package_data={"": ["*.yaml", "*.yml", "*.sh"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    license_files=["LICENSE"],
)
