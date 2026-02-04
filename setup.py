"""
Setup script for the Viral-Local package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open("requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="viral-local",
    version="0.1.0",
    author="Viral-Local Team",
    author_email="team@viral-local.com",
    description="Automated video localization pipeline for YouTube content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/viral-local/viral-local",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Localization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.88.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "streamlit-option-menu>=0.3.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "viral-local=viral_local.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "viral_local": ["*.yaml", "*.yml", "*.json"],
    },
)