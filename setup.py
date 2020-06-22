import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchseq", # Replace with your own username
    version="0.0.1",
    author="Tom Hosking",
    author_email="code@tomho.sk",
    description="A Seq2Seq framework for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomhosking/torchseq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    entry_points = {
        'console_scripts': ['torchseq=torchseq.main:run'],
    },
)