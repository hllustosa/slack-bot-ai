from __future__ import annotations

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field


class Event(BaseModel):
    class Config:
        extra = Extra.allow

    type: str = Field(..., title='The specific name of the event')
    event_ts: str = Field(..., title='When the event was dispatched')


class SlackEvent(BaseModel):
    class Config:
        extra = Extra.allow

    token: str = Field(
        ..., title='A verification token to validate the event originated from Slack',
    )
    team_id: str = Field(
        ...,
        examples=['T1H9RESGL'],
        title='The unique identifier of the workspace where the event occurred',
    )
    api_app_id: str = Field(
        ...,
        description=' Use this to distinguish which app the event belongs to if'
        ' you use multiple apps with the same Request URL.',
        examples=['A2H9RFS1A'],
        title='The unique identifier your installed Slack application.',
    )
    event: Event = Field(
        ...,
        examples=[
            {
                'type': 'message',
                'user': 'U061F7AUR',
                'text': 'How many cats did we herd yesterday?',
                'ts': '1525215129.000001',
                'channel': 'D0PNCRP9N',
                'event_ts': '1525215129.000001',
                'channel_type': 'app_home',
            },
        ],
        title='The actual event, an object, that happened',
    )
    type: str = Field(
        ...,
        examples=['event_callback'],
        title='Indicates which kind of event dispatch this is, usually `event_callback`',
    )
    event_id: str = Field(
        ...,
        examples=['Ev0PV52K25'],
        title='A unique identifier for this specific event, globally unique across all workspaces.',
    )
    event_time: int = Field(
        ...,
        examples=[1525215129],
        title='The epoch timestamp in seconds indicating when this event was dispatched.',
    )
    authed_users: list[str] = Field(
        ...,
        min_items=1,
        title='An array of string-based User IDs. Each member of the collection represents a '
        'user that has installed your application/bot and indicates the described event '
        'would be visible to those users.',
    )
