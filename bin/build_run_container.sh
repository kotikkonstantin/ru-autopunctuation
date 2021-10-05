#!/usr/bin/env bash
docker build -t $2 .
docker run -it -p $1:80 \
  --name=$2 \
  -v $PWD:/app $2
