from setuptools import setup, find_packages

with open("./requirements.txt", "rt") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="calibromatic",
    install_requires=install_requires,
    packages=find_packages()
)
