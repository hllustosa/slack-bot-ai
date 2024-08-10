from __future__ import annotations

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from project.data.vector_store import VectorStore
from project.llm.prompting import QNA_PROMPT


class QueryConfluenceUseCase:
    def __init__(self, llm: ChatOpenAI):
        self.LLM = llm

    @staticmethod
    def run_as_tool(query: str, llm: ChatOpenAI):
        return QueryConfluenceUseCase(llm=llm).execute(query)

    def execute(self, query: str):
        db = VectorStore()
        retriever = db.get_store().as_retriever()

        def format_docs(docs):
            return '\n\n'.join(
                f"{doc.page_content}\nsource:{doc.metadata.get('source')}"
                for doc in docs
            )

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.LLM,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': QNA_PROMPT},
        )

        input = {'question': query, 'chat_history': []}
        retrieved = qa.invoke(input=input)
        sources = list(
            {
                doc.metadata.get('source')
                for doc in retrieved['source_documents']
            },
        )
        answer = retrieved['answer']

        refine_prompt_messages = [
            (
                'system',
                "Based on the context provided, refine the answer to the user's question, "
                "which is about Vinta's internal documentation. Include source links whenever possible.",
            ),
            (
                'human',
                f"context: {format_docs(retrieved['source_documents'])} \n\n answer to user question:{answer}",
            ),
        ]

        refined = self.LLM.invoke(refine_prompt_messages)

        return {'answer': refined.content, 'links': sources}
