# ---------------------------
# FIRST STAGE
# ---------------------------
FROM tensorflow/tensorflow:1.10.1-gpu-py3 AS builder

MAINTAINER Andrzej Pronobis "a@pronobis.pro"

# Username and password for accessing github repo
ARG USERNAME
ARG PASSWORD

# Install missing packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ssh-client

# Clone libspn dev branch using username and password
RUN git clone "https://$USERNAME:$PASSWORD@github.com/pronobis/libspn.git" -b dev /root/libspn

# Change remote to ssh so that password is not visible in second stage
RUN cd /root/libspn && git remote set-url origin git@github.com:pronobis/libspn.git

# Remove logs since they keep old url
RUN rm -r /root/libspn/.git/logs


# ---------------------------
# FINAL STAGE
# ---------------------------
FROM tensorflow/tensorflow:1.10.1-gpu-py3 AS finalstage
WORKDIR "/root"

# Install missing packages
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
ssh-client \
python3-tk

# Get libspn repo from first stage
COPY --from=builder /root/libspn /root/libspn

# Install libspn
# Need to use stub for libcuda.so (https://github.com/NVIDIA/nvidia-docker/issues/374)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    cd libspn && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    make dev-install && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
