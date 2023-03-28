.PHONY: lint test cov

all: lint test

lint:
	isort .
	black .

test:
	pytest test

cov:
	pytest test --cov=tinynn --cov-report term-missing
