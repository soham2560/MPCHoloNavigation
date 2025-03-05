FROM ghcr.io/soham2560/humble-garden:latest

# Add the following labels
LABEL org.opencontainers.image.description="ROS2 Humble + ROS2 Garden + ROS2 Control"
LABEL org.opencontainers.image.title="humble-garden-control"
LABEL org.opencontainers.image.vendor="Soham Patil"
LABEL org.opencontainers.image.source="https://github.com/soham2560/DockerForROS2Development"
LABEL maintainer="sohampatil45939@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

# handle default shell
SHELL ["/bin/bash", "-c"]

COPY rosPkgs.list /tmp/rosPkgs.list
RUN sudo apt-get update \
   && sudo apt-get -y install --no-install-recommends $(cat /tmp/rosPkgs.list) \
   && sudo apt-get autoremove -y \
   && sudo apt-get clean -y \
   && sudo rm -rf /var/lib/apt/lists/*