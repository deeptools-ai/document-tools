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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


DOCUMENT_EXTENSIONS = [
    "ai",
    "bmp",
    "eps",
    "gif",
    "jpg",
    "jpeg",
    "pdf",
    "png",
    "psd",
    "raw",
    "svg",
    "tif",
    "tiff",
    "webp",
]


@dataclass
class Document:
    """
    Base class for all documents.

    Parameters
    ----------
    path : str or Path
        Path to the document.

    Attributes
    ----------
    file : str
        Name of the document.
    extension : str
        Extension of the document.

    Raises
    ------
    ValueError
        If the extension of the document is not valid. Valid extensions are ai, bmp, eps, gif, jpg, jpeg, pdf, png, psd,
        raw, svg, tif, tiff and webp.

    Examples
    --------
    >>> from document_tools.documents.base import Document
    >>> document = Document("my_image.jpg")
    >>> document.file
    'my_image.jpg'
    >>> document.extension
    'jpg'
    """

    _path: Union[str, Path] = field(repr=False)
    file: str = field(init=False)
    extension: str = field(init=False)

    def __post_init__(self):
        if isinstance(self._path, str):
            self._path = Path(self._path)

        self.file = self._path.name

        self.extension = self.file.split(".")[-1]
        if self.extension not in DOCUMENT_EXTENSIONS:
            raise ValueError(
                f"{self.extension} is not a valid extension. Valid extensions are: {', '.join(DOCUMENT_EXTENSIONS)}"
            )


@dataclass
class ImageDocument(Document):
    """
    Class for image documents.
    """

    def __post_init__(self):
        super().__post_init__()


@dataclass
class PDFDocument(Document):
    """
    Class for pdf documents.
    """

    def __post_init__(self):
        super().__post_init__()
