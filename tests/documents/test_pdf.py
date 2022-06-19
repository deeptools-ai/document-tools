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

from document_tools.documents.base import PDFDocument


@pytest.fixture
def correct_paths():
    return ["file.pdf", "my_favorite_file.pdf", "my.other.file.pdf", "/home/user/file.pdf", "home/username/file.pdf"]


@pytest.fixture
def incorrect_paths():
    return ["file1.txt", "/home/user/file.txt", "Desktop/document.sgi", "user.document.pcx"]


def test_pdf_document_class(correct_paths: List[str], incorrect_paths: List[str]):
    """
    Test the PDFDocument class.
    """
    for path in correct_paths:
        document = PDFDocument(path)
        assert document._path == Path(path)
        assert document.file == path.split("/")[-1]
        assert document.extension == path.split(".")[-1]

    for path in incorrect_paths:
        with pytest.raises(ValueError):
            PDFDocument(path)


def test_pdf_document_repr(correct_paths: List[str]):
    """
    Test the PDFDocument class repr.
    """
    for path in correct_paths:
        document = PDFDocument(path)
        assert repr(document) == f"PDFDocument(file='{path.split('/')[-1]}', extension='{path.split('.')[-1]}')"


def test_pdf_document_eq(correct_paths: List[str]):
    """
    Test the equality of two documents.
    """
    for path in correct_paths:
        document1 = PDFDocument(path)
        document2 = PDFDocument(path)
        assert document1 == document2
