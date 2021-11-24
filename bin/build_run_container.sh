#!/usr/bin/env bash
docker build --build-arg CONTAINER_EXPOSED_PORT=$2 -t $3 .
docker run -it -p $1:$2 \
  --name=$3 \
  -v $PWD:/app $3
