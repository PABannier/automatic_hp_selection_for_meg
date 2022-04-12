from setuptools import setup, find_packages


setup(
    name="oracle_calibration",
    install_requires=[
        "numpy>=1.12",
        "joblib",
        "celer",
        "mne",
        "scikit-learn>=0.23",
        "matplotlib>=2.0.0",
    ],
    packages=find_packages()
)
