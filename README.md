# Document Tools


[![pypi](https://img.shields.io/pypi/v/document-tools.svg)](https://pypi.org/project/document-tools/)
[![python](https://img.shields.io/pypi/pyversions/document-tools.svg)](https://pypi.org/project/document-tools/)
[![Build Status](https://github.com/deeptools-ai/document-tools/actions/workflows/dev.yml/badge.svg)](https://github.com/deeptools-ai/document-tools/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/deeptools-ai/document-tools/branch/main/graphs/badge.svg)](https://codecov.io/github/deeptools-ai/document-tools)



🔧 Tools to automate your document understanding tasks.

This package contains tools to automate your document understanding tasks by leveraging the power of
[🤗 Datasets](https://github.com/huggingface/datasets) and [🤗 Transformers](https://github.com/huggingface/transformers).

With this package, you can (or will be able to):

- 🚧 **Create** a dataset from a collection of documents.
- ✅ **Transform** a dataset to a format that is suitable for training a model.
- 🚧 **Train** a model on a dataset.
- 🚧 **Evaluate** the performance of a model on a dataset of documents.
- 🚧 **Export** a model to a format that is suitable for inference.


## Features

This project is under development and is in the alpha stage. It is not ready for production use, and if you find any
bugs or have any suggestions, please let us know by opening an [issue](https://github.com/deeptools-ai/document-tools/issues)
or a [pull request](https://github.com/deeptools-ai/document-tools/pulls).

### Featured models

- [ ] [DiT](https://huggingface.co/docs/transformers/model_doc/dit)
- [x] [LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)
- [x] [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
- [ ] [LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)

## Usage

One-liner to get started:

```python
from datasets import load_dataset
from document_tools import tokenize_dataset

# Load a dataset from 🤗 Hub
dataset = load_dataset("deeptools-ai/test-document-invoice", split="train")

# Tokenize the dataset
tokenized_dataset = tokenize_dataset(dataset, target_model="layoutlmv3")
```

For more information, please see the [documentation](https://deeptools-ai.github.io/document-tools/)

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
