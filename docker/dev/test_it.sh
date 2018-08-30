#!/bin/sh

# Forward ssh agent
nvidia-docker run -it --rm \
              -v $SSH_AUTH_SOCK:/.ssh-agent -e SSH_AUTH_SOCK=/.ssh-agent \
              pronobis/libspn:dev-latest-gpu bash
