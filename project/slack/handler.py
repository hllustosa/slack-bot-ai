from __future__ import annotations

import logging
import os

from slack_bolt import App
from slack_bolt import Say
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slackstyler import SlackStyler

from project.llm.gpt import GPT_4o_MINI as LLM
from project.slack.schemas import SlackEvent
from project.use_cases.answer_slack_message_with_llm import AnswerSlackMessageUseCase

logging.basicConfig(level=logging.INFO)

styler = SlackStyler()
app = App(
    token=os.environ.get('SLACK_BOT_TOKEN'),
    signing_secret=os.environ.get('SLACK_SIGNING_SECRET'),
)


def truncate(text: str, max_len=50):
    if len(text) > max_len:
        return text[0:max_len] + '...'
    return text

# avoid looping on self messages


def no_bot_messages(message: SlackEvent, next):
    subtype = message.get('subtype') if message else ''
    if subtype != 'bot_message':
        next()


@app.event('app_mention', middleware=[no_bot_messages])
def handle_app_mentions(body: SlackEvent, say: Say, logger: logging.Logger):
    mention = body.get('event')
    user, text = (mention['user'], mention['text'])

    parsed_mention_text = text.replace(r'<@\w+>', '').strip()
    logger.info(f'New mention from {user}: {truncate(parsed_mention_text)}')

    _, *ai_messages = AnswerSlackMessageUseCase(
        llm_with_tools=LLM,
    ).execute(parsed_mention_text)

    for message in ai_messages:
        logger.info('AI Generated Message: %s', truncate(message.content))
        if message.content:
            say(styler.convert(message.content))


@app.event('message', middleware=[no_bot_messages])
def handle_message(message: SlackEvent, say: Say, logger: logging.Logger):
    user, text = (message['user'], message['text'])
    logger.info(f'New message from {user}: {text}')
    _, * \
        ai_messages = AnswerSlackMessageUseCase(
            llm_with_tools=LLM,
        ).execute(text)

    for message in ai_messages:
        logger.info('AI Generated Message: %s', truncate(message.content))
        if message.content:
            say(styler.convert(message.content))


slack_handler = SlackRequestHandler(app)
