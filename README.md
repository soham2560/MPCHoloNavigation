# MPCHoloNavigation
ROS2 Setup to perform MPC based Holonomic Navigation for Robotics: Planning and Navigation Course (EC4.403) at IIITH

- Professor: [K. Madhava Krishna](https://faculty.iiit.ac.in/~mkrishna/)

## Table of contents

- [LiDAR Camera Calibration](#lidar-camera-calibration)
  - [Table of contents](#table-of-contents)
  - [Docker Setup](#docker-setup)


## Docker Setup
- To pull latest docker image
    ```bash
    docker pull ghcr.io/soham2560/humble-garden:latest
    ```
- To start container
    - Open Command Pallete with `Ctrl+Shift+P`
    - Select Option to Rebuild and Reopen Container
    - Use `Build WS` button to build workspace
  
  Note: To access these buttons you may need to enable [VSCode Action Button Extension](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons) through the Extensions Tab in VSCode, the extension should download automatically on container startup
  

Note: The README's in this repository are inspired by [this](https://github.com/TheProjectsGuy/MR21-CS7.503) and [this](https://github.com/ankitdhall/lidar_camera_calibration)