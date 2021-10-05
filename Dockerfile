FROM debian:latest
RUN apt update && apt upgrade -y && \
    apt install -y python3 g++ make python3-pip curl git nano

WORKDIR /.

COPY './requirements.txt' .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Disable huggungface tokenizers parallelism
ENV TOKENIZERS_PARALLELISM "false"

ENV TASK_COMMAND "serve"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
#CMD ["python3", "app.py" ]