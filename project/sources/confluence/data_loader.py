from __future__ import annotations

import os

from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DataLoader:
    """Create, load, save the DB using the confluence Loader"""

    def __init__(self, **kwargs):
        self.username = os.getenv('CONFLUENCE_USER')
        self.api_key = os.getenv('CONFLUENCE_TOKEN')
        self.base_url = os.getenv('CONFLUENCE_ROOT')
        self.extra_loader_kwargs = kwargs

    def load(self, space_key=None, **kwargs) -> list[Document]:
        loader = ConfluenceLoader(
            url=self.base_url,
            username=self.username,
            api_key=self.api_key,
            space_key=space_key,
            limit=50,
        )

        return loader.load()

    def split_docs(self, docs: list[Document]) -> list[Document]:
        headers_to_split_on = [
            ('#', 'H1'),
            ('##', 'H2'),
            ('###', 'H3'),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )

        md_docs = []

        for doc in docs:
            md_doc = markdown_splitter.split_text(doc.page_content)
            for i in range(len(md_doc)):
                md_doc[i].metadata = md_doc[i].metadata | doc.metadata
            md_docs.extend(md_doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=20,
            separators=['\n\n', '\n', r'(?<=\. )', ' ', ''],
        )

        pre_processed_docs = splitter.split_documents(md_docs)
        return pre_processed_docs
