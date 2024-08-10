from __future__ import annotations

from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate


system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If there are links to sources in the metadata, include them in your answer. The questions are
always about Vinta's (Software Product Development Company) internal documentation. The context
provided to you is the most relevant information to answer the question, based on Vinta's knowledge
base and the query. If you need more context, ask the user for more information. Send your answer
as slack flavored markdown.
----------------
{context}"""

human_template = '{question}'

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
]


QNA_PROMPT = ChatPromptTemplate.from_messages(messages)
