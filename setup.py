from setuptools import setup, find_packages


setup(
    name="hp_selection",
    install_requires=[
        "numpy>=1.12",
        "joblib",
        "mne",
        "scikit-learn>=0.23",
        "matplotlib>=2.0.0",
    ],
    packages=find_packages()
)
