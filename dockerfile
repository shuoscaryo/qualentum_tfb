FROM debian:12.6

RUN apt-get update && apt-get install -y \
	python3 \
	pip

RUN mkdir /app

RUN echo 'PS1="\[\][docker:\u] \W \[\e[1m\]>>\[\e[0m\] \[\]"' >> /etc/bash.bashrc

WORKDIR /app

CMD ["bash"]