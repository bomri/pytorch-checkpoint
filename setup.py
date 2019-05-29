import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorchcheckpoint",
    version="0.0.4",
    author="Omri Bar",
    author_email="baromri@gmail.com",
    description="Support PyTorch checkpoints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bomri/pytorch-checkpoint.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
