docker_build:
	@docker compose up --no-start slackbot-ai

docker_start_all: docker_build
	@docker compose start slackbot-ai

docker_stop_all:
	@docker compose stop

lint:
	poetry run pre-commit install && poetry run pre-commit run -a -v
