from __future__ import annotations

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Query
from fastapi import Request

from project.llm.gpt import GPT_4o_MINI as LLM
from project.llm.gpt import tools
from project.slack.handler import slack_handler
from project.use_cases.ingest_confluence_space import IngestConfluenceSpaceUseCase
from project.use_cases.query_confluence import QueryConfluenceUseCase

load_dotenv()


tags_metadata = [
    {
        'name': 'Info',
        'description': 'Root endpoint to show info about the API',
    },
    {
        'name': 'Confluence Integration',
        'description': 'Uses built-in Confluence Loader to ingest and query '
        '[Confluence Spaces with LangChain]'
        '(https://api.python.langchain.com/en/latest/document_loaders/'
        'langchain_community.document_loaders.confluence.ConfluenceLoader.html)',
    },
    {
        'name': 'Slack Events',
        'description': "Receives Slack Webhook Payloads with [Slack's Library](https://slack.dev/bolt-python/)",
        'externalDocs': {
            'description': 'Slack Events API',
            'url': 'https://api.slack.com/apis/events-api',
        },
    },
]

app = FastAPI(
    title='Slack Assistant API',
    version='0.1.0',
    summary='Endpoints to manage Confluence RAG and Slack Webhook Events',
    openapi_tags=tags_metadata,
)


@app.get('/', tags=['Info'])
def show_info():
    return {tool.func.__name__: tool.func.__doc__ for tool in tools}


@app.post('/confluence/{space_key}/', tags=['Confluence Integration'])
def ingest_knowledge_base(space_key: str):
    return IngestConfluenceSpaceUseCase(space_key=space_key).execute()


@app.get('/confluence/', tags=['Confluence Integration'])
def query_knowledge_base(query: str = Query(...)):
    return QueryConfluenceUseCase(llm=LLM).execute(query)


@app.post('/slack/events/', tags=['Slack Events'])
async def slack_webhook(req: Request):
    challenge = req.get('challenge')

    if challenge:  # dummy implementation
        return {'challenge': challenge}

    return await slack_handler.handle(req)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7999)
