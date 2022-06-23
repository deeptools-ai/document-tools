# Usage

To use Document Tools in a project

```python
from datasets import load_dataset
from document_tools import tokenize_dataset

# Load a dataset from 🤗 Hub
dataset = load_dataset("deeptools-ai/test-document-invoice", split="train")

# Tokenize the dataset
tokenized_dataset = tokenize_dataset(dataset, target_model="layoutlmv3")
```

You can also save the tokenized dataset to 🤗 Hub:

```python
tokenized_dataset.push_to_hub("YourName/YourProject")
```

Or save it directly to your local machine as a 🤗 Dataset:

```python
tokenized_dataset = tokenize_dataset(dataset, target_model="layoutlmv3", save_to_disk=True, save_path="path/to/save/to")
```

Learn more about the available parameters for `tokenize_dataset` in the [documentation](./api.md)
