# coding=utf-8
#
# Copyright The deeptools.ai team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
from datasets import DatasetDict, load_dataset

from document_tools import TARGET_MODELS, tokenize_dataset


@pytest.fixture
def incorrect_dataset_format():
    """Return a list of incorrect dataset formats."""
    return [
        "1, 2, 3, 4, 5",
        (1, 2, 3, 4, 5),
        [1, 2, 3, 4, 5],
        np.array([1, 2, 3, 4, 5]),
        {"train": 1, "valid": 2, "test": 3},
        pd.DataFrame.from_dict({"train": [1, 2, 3], "valid": [1, 2, 3], "test": [1, 2, 3]}),
    ]


@pytest.fixture
def dataset_for_testing():
    """Return a dataset for testing."""
    dataset = load_dataset("deeptools-ai/test-document-invoice", download_mode='force_redownload')
    dataset = dataset["train"].select(range(2))
    return dataset


def test_incorrect_dataset_format(incorrect_dataset_format: List[Any]):
    """Test that the function raises an error when the dataset is not in the correct format."""
    for dataset in incorrect_dataset_format:
        with pytest.raises(TypeError):
            tokenize_dataset(dataset, target_model="layoutlmv2")


def test_correct_dataset_format(dataset_for_testing: DatasetDict):
    """Test that the function accepts the correct dataset format."""
    tokenize_dataset(dataset_for_testing, target_model="layoutlmv2")


def test_correct_dataset_dict_format(dataset_for_testing: DatasetDict):
    """Test that the function accepts the correct dataset format."""
    dataset_dict = DatasetDict()
    dataset_dict["train"] = dataset_for_testing
    tokenize_dataset(dataset_dict, target_model="layoutlmv2")


def test_target_models(dataset_for_testing: DatasetDict):
    """Test that the function accepts the correct target model."""
    tokenize_dataset(dataset_for_testing, target_model="layoutlmv2")


def test_incorrect_target_models(dataset_for_testing: DatasetDict):
    """Test that the function raises an error when the target model is not available."""
    model = "layoutlmv2"
    model = model[::-1]
    with pytest.raises(KeyError):
        tokenize_dataset(dataset_for_testing, target_model=model)


def test_save_method_without_path(dataset_for_testing: DatasetDict):
    """Test that the function raises an error when the save_to_disk is True and the save_path is not provided."""
    with pytest.raises(ValueError):
        tokenize_dataset(dataset_for_testing, target_model="layoutlmv2", save_to_disk=True)


def test_save_method_is_false_with_path(caplog, dataset_for_testing: DatasetDict):
    """Test that the function raises an error when the save_to_disk is False and the save_path is provided."""
    tokenize_dataset(dataset_for_testing, target_model="layoutlmv2", save_to_disk=False, save_path="/home/code")
    assert caplog.records[0].levelname == "WARNING"
    assert (
        """
            You have indicated a path to save the dataset, but have chosen not to save it to disk. You need to add
            `save_to_disk=True` to the call to `tokenize_dataset` to save the dataset to disk.
        """
        in caplog.text
    )


def test_log_if_save_to_disk_is_true(caplog, dataset_for_testing: DatasetDict):
    """Test that the function logs if the save_to_disk is True."""
    tokenize_dataset(dataset_for_testing, target_model="layoutlmv2", save_to_disk=True, save_path="/home/code")
    assert caplog.records[0].levelname == "ERROR"


def test_target_models_metadata():
    """Test that the function returns the correct metadata for the target models."""
    assert len(list(TARGET_MODELS.keys())) == 3
    assert list(TARGET_MODELS.keys()) == ["layoutlmv2", "layoutlmv3", "layoutxlm"]
    assert TARGET_MODELS["layoutlmv2"].__name__ == "LayoutLMv2Encoder"
    assert TARGET_MODELS["layoutlmv3"].__name__ == "LayoutLMv3Encoder"
    assert TARGET_MODELS["layoutxlm"].__name__ == "LayoutXLMEncoder"


def test_without_target_model(dataset_for_testing: DatasetDict):
    """Test that the function raises an error when the target model is not provided."""
    with pytest.raises(ValueError):
        tokenize_dataset(dataset_for_testing)


def test_target_model_is_a_string(dataset_for_testing: DatasetDict):
    """Test that the function raises an error when the target model is not a string."""
    with pytest.raises(KeyError):
        tokenize_dataset(dataset_for_testing, target_model=1)  # type: ignore


def test_layout_lmv2(dataset_for_testing: DatasetDict):
    """Test that the function returns the correct metadata for the target models."""
    tmp = tokenize_dataset(dataset_for_testing, target_model="layoutlmv2")
    print(tmp)  # TODO: check that the tokenized dataset is correct


def test_layout_lmv3(dataset_for_testing: DatasetDict):
    """Test that the function returns the correct metadata for the target models."""
    tmp = tokenize_dataset(dataset_for_testing, target_model="layoutlmv3")
    print(tmp)  # TODO: check that the tokenized dataset is correct


def test_layout_xlm(dataset_for_testing: DatasetDict):
    """Test that the function returns the correct metadata for the target models."""
    tmp = tokenize_dataset(dataset_for_testing, target_model="layoutxlm")
    assert tmp is not None
    tmp_train = tmp["train"]
    assert len(tmp_train) == 2

    assert type(tmp_train["input_ids"]) is list
    assert len(tmp_train["input_ids"][0]) == 155
    assert len(tmp_train["input_ids"][1]) == 228

    assert type(tmp_train["attention_mask"]) is list
    assert len(tmp_train["attention_mask"][0]) == 155
    assert len(tmp_train["attention_mask"][1]) == 228

    assert type(tmp_train["labels"]) is list
    assert tmp_train["labels"][0] == [6]
    assert len(tmp_train["labels"][0]) == 1
    assert tmp_train["labels"][1] == [6]
    assert len(tmp_train["labels"][1]) == 1
