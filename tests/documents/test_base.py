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
from typing import List

import pytest

from document_tools.documents.base import Document


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
        "my_image.raw" "my_image.svg",
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
    return ["file1.txt", "/home/user/file.txt", "Desktop/document.sgi", "user.document.pcx"]


def test_document_base_class(correct_paths: List[str], incorrect_paths: List[str]):
    """
    Test the base class for all documents.
    """
    for path in correct_paths:
        document = Document(path)
        assert document._path == Path(path)
        assert document.file == path.split("/")[-1]
        assert document.extension == path.split(".")[-1]

    for path in incorrect_paths:
        with pytest.raises(ValueError):
            Document(path)


def test_document_repr(correct_paths: List[str]):
    """
    Test the representation of a document.
    """
    for path in correct_paths:
        document = Document(path)
        assert repr(document) == f"Document(file='{path.split('/')[-1]}', extension='{path.split('.')[-1]}')"


def test_document_eq(correct_paths: List[str]):
    """
    Test the equality of two documents.
    """
    for path in correct_paths:
        document1 = Document(path)
        document2 = Document(path)
        assert document1 == document2
