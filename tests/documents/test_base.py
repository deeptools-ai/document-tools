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
from pathlib import Path
from typing import Any, List

import pytest

from document_tools.documents import BaseDocument


@pytest.fixture
def correct_paths():
    return [
        "my_image.ai",
        "my_image.bmp",
        "my_image.eps",
        "my_image.gif",
        "my_image.jpg",
        "my_image.jpeg",
        "my_image.pdf",
        "my_image.png",
        "my_image.psd",
        "my_image.raw",
        "my_image.svg",
        "my_image.tif",
        "my_image.tiff",
        "my_image.webp",
        "my_favorite_file.pdf",
        "my.other.file.pdf",
        "/home/user/file.jpeg",
        "home/username/file.tiff",
    ]


@pytest.fixture
def incorrect_paths():
    return ["", "file.txt", "/home/user/file.txt", "Desktop/document.sgi", "user.document.pcx"]


@pytest.fixture
def incorrect_objects():
    return [
        1,
        True,
        False,
        None,
        [],
        (),
        {},
        BaseDocument("/home/user/file.png"),
        BaseDocument("/home/user/file.pdf"),
    ]


@pytest.fixture
def tokenizer():
    return ["layoutlm", "layoutlmv2", "layoutlmv3", "layoutxlm"]


def test_document_base_class(correct_paths: List[str], incorrect_paths: List[str], incorrect_objects: List[Any]):
    """Test the base class for all documents."""
    for path in correct_paths:
        document = BaseDocument(path)
        assert document._path == Path(path)
        assert document.file == path.split("/")[-1]
        assert document.extension == path.split(".")[-1]

    for path in incorrect_paths:
        with pytest.raises(ValueError):
            BaseDocument(path)

    for item in incorrect_objects:
        with pytest.raises(TypeError):
            BaseDocument(item)


def test_document_repr(correct_paths: List[str]):
    """Test the representation of a document."""
    for path in correct_paths:
        document = BaseDocument(path)
        assert repr(document) == f"BaseDocument(file='{path.split('/')[-1]}', extension='{path.split('.')[-1]}')"


def test_document_eq(correct_paths: List[str]):
    """Test the equality of two documents."""
    for path in correct_paths:
        document1 = BaseDocument(path)
        document2 = BaseDocument(path)
        assert document1 == document2


def test_load_method():
    """Test the load method of a document."""
    with pytest.raises(NotImplementedError):
        BaseDocument("/home/user/file.png").load()


def test_tokenize_method(tokenizer: List[str]):
    """Test the tokenize method of a document."""
    for tokenizer_name in tokenizer:
        with pytest.raises(NotImplementedError):
            BaseDocument("/home/user/file.png").tokenize(tokenizer_name)


def test_save_method():
    """Test the save method of a document."""
    with pytest.raises(NotImplementedError):
        BaseDocument("/home/user/file.png").save()
