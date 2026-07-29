from setuptools import find_packages, setup

setup(
    name="moonep-td",
    version="0.1.0",
    packages=find_packages(include=["moonep_td", "moonep_td.*"]),
    python_requires=">=3.11",
    install_requires=["torch>=2.6,<2.7"],
)
