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
from document_tools.documents.base import Document


def test_document_base_class():
    """
    Test the base class for all documents.
    """
    document = Document("test.txt")
    assert document.file == "test.txt"
    assert document.extension == "txt"


def test_document_repr():
    """
    Test the representation of a document.
    """
    document = Document("test.txt")
    assert repr(document) == "Document(file='test.txt')"


def test_document_str():
    """
    Test the string representation of a document.
    """
    document = Document("test.txt")
    assert str(document) == "test.txt"
