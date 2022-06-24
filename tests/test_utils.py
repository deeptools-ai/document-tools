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

from document_tools.utils import _get_label_list


@pytest.fixture()
def list_of_labels_with_duplicate_strings():
    """List of labels with duplicate strings."""
    return [["bill", "invoice", "payment", "receipt", "bill", "invoice", "bill"]]


@pytest.fixture()
def list_of_labels_with_duplicate_integers():
    """List of labels with duplicate integers."""
    return [[1, 2, 3, 4, 5, 2, 3, 3, 2]]


@pytest.fixture()
def list_of_wrong_labels():
    return [
        {"bill": 1, "invoice": 2, "payment": 3, "receipt": 4},
        (1, 2, 3, 4),
        np.array([1, 2, 3, 4]),
        pd.DataFrame.from_dict({"bill": [1, 2, 3], "invoice": [1, 2, 3], "payment": [1, 2, 3], "receipt": [1, 2, 3]}),
    ]


def test_get_label_list_with_duplicate(list_of_labels_with_duplicate_strings: List[str]):
    """Test that the function returns the correct list of labels."""
    label_list = _get_label_list(list_of_labels_with_duplicate_strings)
    assert label_list == ["bill", "invoice", "payment", "receipt"]


def test_get_label_list_with_duplicate_integers(list_of_labels_with_duplicate_integers: List[int]):
    """Test that the function returns the correct list of labels."""
    label_list = _get_label_list(list_of_labels_with_duplicate_integers)
    assert label_list == [1, 2, 3, 4, 5]


def test_get_label_list_wrong_input(list_of_wrong_labels: List[Any]):
    """Test that the function raises an error when the input is not a list of labels."""
    for label_list in list_of_wrong_labels:
        with pytest.raises(TypeError):
            _get_label_list(label_list)
