from setuptools import find_packages, setup

setup(
    name="hdgp-vlm",
    version="0.1.0",
    description="Vision-grounded hierarchical pouring orchestration",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
