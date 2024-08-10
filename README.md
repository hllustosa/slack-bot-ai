# Micro Slack Assistant

This is a placeholder, further documentation will be more detailed.

## Getting Started

See .env.example and project/api to get an overview of what this service does. Currently available
tools are mostly implemented according to LangChain examples

## Goals

- As a Developer, I'd like to be able to deploy a Slack assistant that can be installed
    in any workspace

- As a User, I'd like to message the bot directly or mention it on a channel, and have
    an answer to my query

- As a User, I'd like my bot to have access to web search and internal documentation, and
    use these resources to answer when appropriate

## TODOs

Most important first:

### Security
- [ ] Proper handling of Slack security (verification token, webhook url challenge)
- [ ] Implement Token Auth for management endpoints

### Deployment
- [ ] Allow dynamic setting of tool docstrings (override source defaults)
- [ ] Check viability of deploying SQLite or change vector store
- [ ] Deploy on Docker with GitHub Actions

### Readability
- [ ] Check LangChain usage for anti-patterns and refactoring opportunities
- [ ] Proper project documentation

### Performance

- [ ] Metrics on used tokens, test lighter models for a few tasks, general cost reduction
- [ ] Viability of LLM inference caching by query semantics


## Possible Features

- [ ] Message history and expanded context for answers
- [ ] Multiple Retrievers (Library docs, community references, RSS Feeds)
- [ ] Slash commands (/tldr, /remindme, /research)
