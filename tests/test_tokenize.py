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

from document_tools import tokenize_dataset
from document_tools.tokenize import TARGET_MODELS


@pytest.fixture
def incorrect_dataset_format():
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
    dataset = load_dataset("deeptools-ai/test-document-invoice")
    dataset = dataset["train"].select(range(2))
    return dataset


@pytest.fixture
def correct_models():
    return list(TARGET_MODELS.keys())


def test_incorrect_dataset_format(incorrect_dataset_format: List[Any], correct_models: List[str]):
    """"""
    for dataset in incorrect_dataset_format:
        with pytest.raises(TypeError):
            tokenize_dataset(dataset, target_model=correct_models[0])


def test_correct_dataset_format(dataset_for_testing: DatasetDict, correct_models: List[str]):
    """"""
    tokenize_dataset(dataset_for_testing, target_model=correct_models[0])
    dataset_dict = DatasetDict()
    dataset_dict["train"] = dataset_for_testing
    tokenize_dataset(dataset_dict, target_model=correct_models[0])


def test_target_models(dataset_for_testing: DatasetDict, correct_models: List[str]):
    """"""
    assert len(correct_models) == 3
    for model in correct_models:
        tokenize_dataset(dataset_for_testing, target_model=model)


def test_incorrect_target_models(dataset_for_testing: DatasetDict, correct_models: List[str]):
    """"""
    for model in correct_models:
        model = model[::-1]
        with pytest.raises(KeyError):
            tokenize_dataset(dataset_for_testing, target_model=model)


def test_save_method_without_path(dataset_for_testing: DatasetDict, correct_models: List[str]):
    """"""
    with pytest.raises(ValueError):
        tokenize_dataset(dataset_for_testing, target_model=correct_models[0], save_to_disk=True)


def test_target_models_metadata(correct_models: List[str]):
    """"""
    for model in correct_models:
        model_keys = list(TARGET_MODELS[model].keys())
        assert model_keys == ["preprocessor", "default_model", "encode_function", "features"]
        assert len(model_keys) == 4
