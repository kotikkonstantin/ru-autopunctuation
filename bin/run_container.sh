#!/usr/bin/env bash
docker run -it -p $1:80 \
  --name=${app} \
  -v $PWD:/app $2
