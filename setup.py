import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchseq",
    version="3.1.0",
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
        'console_scripts': ['torchseq=torchseq.main:main', 'torchseq-eval=torchseq.eval.cli:main'],
    },
    install_requires = [
        'tensorboard==2.15.0',
        'torch==2.2.2',
        'tqdm>=4.62',
        'scipy>=1.5',
        'nltk>=3.6.7',
        'transformers==4.39.2',
        'tokenizers==0.15.2',
        'jsonlines>=2',
        'sacrebleu>=2.0',
        'py-rouge',
        'rouge-score',
        'wandb==0.15.12',
        'matplotlib',
        'opentsne',
        'sentencepiece==0.1.95',
        'protobuf<4',
        'pydantic==1.10.13',
        'truecase==0.0.14',
        'lightning==2.1.0',
        # 'summac',
        'compress_json==1.0.10',
    ],
)