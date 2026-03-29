#!/bin/bash

NETWORK_NAME=n3xt_net

if ! docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
  echo "Creating network: $NETWORK_NAME"
  docker network create $NETWORK_NAME
else
  echo "Network already exists: $NETWORK_NAME"
fi
