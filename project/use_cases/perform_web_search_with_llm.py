from __future__ import annotations

import json
import os

from langchain_community.tools import BraveSearch
from markdownify import markdownify as md

BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY')


class PerformWebSearchUseCase:
    @staticmethod
    def run_as_tool(query: str):
        raw_results = BraveSearch.from_api_key(
            api_key=BRAVE_SEARCH_API_KEY, search_kwargs={'count': 3},
        ).run(query)
        json_results = json.loads(raw_results)
        html_formatted_answer = ''

        for result in json_results:
            title, link, snippet = (
                result['title'], result['link'], result['snippet'],
            )
            formatted_result = f"""
            <strong>{title}</strong>
            <p>{snippet}</p>
            <a href="{link}">{link}</a>
            """

            html_formatted_answer += f'{formatted_result}\n\n'

        return md(html_formatted_answer)
