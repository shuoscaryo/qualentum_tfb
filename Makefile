WORKDIR := $(shell dirname $(realpath $(MAKEFILE_LIST)))
VOLUME := $(WORKDIR)/src
NAME := ia
DOCKER_RUN := docker run -d -it --name "$(NAME)" -v "$(VOLUME)":/app "$(NAME)"
DOCKER_BUILD := docker build -t "$(NAME)" .

all: check-docker $(VOLUME) stop
	@if [ ! -z "$(shell docker images -q $(NAME))" ]; then \
		echo "Image $(NAME) already exists. Aborting."; \
		echo "Run 'make re' to rebuild the image."; \
		exit 1; \
	fi
	@if [ ! -z "$(shell docker ps -a -q -f name=$(NAME))" ]; then \
		echo "Removing container $(NAME)"; \
		docker rm -f $(NAME); \
	fi

	@echo "Building Docker image $(NAME)"
	$(DOCKER_BUILD) > /dev/null
	@echo "Running Docker container $(NAME)"
	$(DOCKER_RUN) > /dev/null
	@echo "Run 'make bash' to open a bash shell in the container"

build: check-docker stop
	@if [ ! -z "$(docker ps -a -q -f name=$(NAME))" ]; then \
		echo "Removing container $(NAME)"; \
		docker rm -f $(NAME); \
	fi

	@echo "Building Docker image $(NAME)"
	docker build -t $(NAME) .

run: check-docker $(VOLUME)
	@if [ ! -z "$(shell docker ps -q -f name=$(NAME))" ]; then \
		echo "Container $(NAME) is already running."; \
	elif [ -z "$(shell docker images -q $(NAME))" ]; then \
		echo "Image $(NAME) does not exist. Run 'make build' to create it."; \
		exit 1; \
	elif [ ! -z "$(shell docker ps -a -q -f name=$(NAME))" ]; then \
		echo "Starting existing Docker container $(NAME)"; \
		docker start $(NAME) > /dev/null; \
	else \
		echo "Running a new Docker container $(NAME)"; \
		docker run -d -it --name $(NAME) -v $(VOLUME):/app $(NAME); \
	fi
	@echo "Run 'make bash' to open a bash shell in the container"


stop: check-docker
	@if [ ! -z "$(shell docker ps -q -f name=$(NAME))" ]; then \
		echo "Stopping container $(NAME)"; \
		docker stop $(NAME) > /dev/null; \
	fi

clean: check-docker stop
	@if [ ! -z "$(shell docker ps -a -q -f name=$(NAME))" ]; then \
		echo "Removing Docker container $(NAME)"; \
		docker rm -f $(NAME) > /dev/null; \
	fi
	@if [ ! -z "$(shell docker images -q $(NAME))" ]; then \
		echo "Removing Docker image $(NAME)"; \
		docker rmi $(NAME) > /dev/null; \
	fi

re: clean build run

bash: check-docker
	@docker exec -it $(NAME) bash

$(VOLUME):
	@echo "Creating volume $(VOLUME)"
	@mkdir -p $(VOLUME) 2>/dev/null

check-docker:
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker is not running. Please start Docker and try again."; \
		exit 1; \
	fi

.PHONY: check-docker
