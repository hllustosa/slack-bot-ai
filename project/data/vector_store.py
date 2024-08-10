from __future__ import annotations

import sqlite3

import sqlite_vss
from langchain_community.vectorstores import SQLiteVSS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
DB_FILE = './vinta.db'
TABLE = 'vinta'


class VectorStore:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model='text-embedding-3-small',
        )

    def as_embeddable_text(self, documents: list[Document]):
        metadata = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]
        return texts, metadata

    def create_db(self, documents: list[Document]) -> SQLiteVSS:
        texts, metadata = self.as_embeddable_text(documents)
        return SQLiteVSS.from_texts(
            texts=texts,
            metadatas=metadata,
            embedding=self.embedding_function,
            table=TABLE,
            db_file=DB_FILE,
        )

    def get_store(self) -> SQLiteVSS:
        connection = sqlite3.connect(DB_FILE, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vss.load(connection)
        connection.enable_load_extension(False)
        # connection = SQLiteVSS.create_connection(db_file=DB_FILE)
        return SQLiteVSS(
            table=TABLE, embedding=self.embedding_function, connection=connection,
        )

    def query_db(self, query: str) -> list[Document]:
        store = self.get_store()
        return store.similarity_search(query, k=3)
