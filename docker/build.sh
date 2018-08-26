#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage ./build.sh <password>"
    echo ""
    echo "Args:"
    echo "  <password> - github password for user libspn-dev"
    exit 1
fi

nvidia-docker build -t pronobis/libspn:dev-latest-gpu \
              --no-cache --rm \
              --build-arg USERNAME=libspn-dev --build-arg PASSWORD="$1" \
              dev

docker image prune
