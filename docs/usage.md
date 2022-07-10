# Usage

## Simple usage

```python
from datasets import load_dataset
from document_tools import tokenize_dataset

# Load a dataset from 🤗 Hub
dataset = load_dataset("deeptools-ai/test-document-invoice", split="train")

# Tokenize the dataset
tokenized_dataset = tokenize_dataset(dataset, target_model="layoutlmv3")
```

## Save the tokenized dataset

You can save the tokenized dataset to 🤗 Hub:

```python
tokenized_dataset.push_to_hub("user_name/user_project")
```

Or save it directly to your local machine as a 🤗 Dataset:

```python
tokenized_dataset = tokenize_dataset(dataset, target_model="layoutlmv3", save_to_disk=True, save_path="path/to/save/to")
```

You can  choose between three different target models:

* `layoutlmv2`: The LayoutLMv2 model.
* `layoutlmv3`: The LayoutLMv3 model.
* `layoutxlm`: The LayoutXLM model.

All these models are from Microsoft document understanding tools.

## Columns names convention

By default, the column names of the input dataset must be `image` for the image content and `label` for the label
column.

You can change this default convention by passing `image_column` and `label_column` argument to the `tokenize_dataset`
function:

```python
tokenized_dataset = tokenize_dataset(
    dataset,
    target_model="layoutlmv3",
    image_column="invoice_images", # Change this by the name of your image column in your input dataset
    label_column="invoice_labels" # Change this by the name of your label column in your input dataset
)
```

## Processor configuration

You can configure the tokenization processor by passing a `processor_config` argument to the `tokenize_dataset`
function. By default, if no `processor_config` is passed, the processor of the target model will use:

```python
{
    "padding": "max_length",
    "truncation": True,
}
```

You can read more about the arguments that can be passed to the processor in the [Processor documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer.__call__).

Learn more about the available parameters for `tokenize_dataset` in the [documentation](./api.md)
