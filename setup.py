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
        'console_scripts': ['torchseq=torchseq.main:main'],
    },
    install_requires = [
        'tensorboard==2.2.2',
        'torch==1.7.0',
        'tqdm>=4.54',
        'scipy>=1.5',
        'nltk==3.5.0',
        'transformers==4.0.0',
        'tokenizers==0.9.4',
        'jsonlines>=1.2',
        'sacrebleu>=1.4'
    ],
)