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
        'tensorboard==2.7.0',
        'torch==1.10.2',
        'tqdm>=4.62',
        'scipy>=1.5',
        'nltk>=3.6.7',
        'transformers==4.16.2',
        'tokenizers==0.10.3',
        'jsonlines>=2',
        'sacrebleu>=2.0',
        'py-rouge',
        'wandb==0.12.10',
        'pytorch-lightning==1.5.9'
    ],
)