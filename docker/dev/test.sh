#!/bin/sh

# Test NVidia driver and LibSPN
nvidia-docker run --rm pronobis/libspn:dev-latest-gpu nvidia-smi && \
nvidia-docker run --rm pronobis/libspn:dev-latest-gpu bash -c "cd libspn; make test"
