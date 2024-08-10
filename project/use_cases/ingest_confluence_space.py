from __future__ import annotations

from project.data.vector_store import VectorStore
from project.sources.confluence.data_loader import DataLoader


class IngestConfluenceSpaceUseCase:
    def __init__(self, space_key: str):
        self.space_key = space_key

    def execute(self):
        data_loader, db = (DataLoader(), VectorStore())

        docs = data_loader.load(space_key=self.space_key)
        processed_docs = data_loader.split_docs(docs)

        db.create_db(processed_docs)
