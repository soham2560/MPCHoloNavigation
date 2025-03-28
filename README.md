# MPCHoloNavigation
ROS2 Setup to perform MPC based Holonomic Navigation for Robotics: Planning and Navigation Course (EC4.403) at IIITH

- Professor: [K. Madhava Krishna](https://faculty.iiit.ac.in/~mkrishna/)

## Table of contents

- [MPCHoloNavigation](#mpcholonavigation)
  - [Table of contents](#table-of-contents)
  - [Development Setup](#development-setup)
    - [Action Buttons](#action-buttons)
  - [Hardware Setup](#hardware-setup)
  - [How to Use](#how-to-use)


## Development Setup
- To pull latest docker image
    ```bash
    docker pull ghcr.io/soham2560/humble-garden:latest
    ```
- To start container
    - Open Command Pallete with `Ctrl+Shift+P`
    - Select Option to Rebuild and Reopen Container

  ### Action Buttons
  We use the [VSCode Action Button Extension](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons) to facilitate development. They are not necessary but certainly do help. To access these buttons you may need to enable it through the Extensions Tab in VSCode, though the extension should download automatically on container startup. The available buttons are as follows:
  - `Build`

    Builds the workspace packages upto `robot_bringup`

    ```bash
    colcon build --symlink-install --packages-up-to robot_bringup
    ```

  - `Import Libs`

    Uses [vcstool](https://github.com/dirk-thomas/vcstool) to import the repositories listed in [dep.repos](dep.repos)

    ```bash
    rm -rf /ros2_ws/src/dep_repos && mkdir -p /ros2_ws/src/dep_repos && vcs import /ros2_ws/src/dep_repos < /ros2_ws/src/dep.repos
    ```

  Note: Remember to use `Import Libs` on every fresh container startup (not necessary on reopen)

## Hardware Setup
  Information about the Hardware Setup can be found [here](/docs/hardware.md)

## How to Use
- **Build and Source the workspace**

  It is suggested to use the [Action Buttons](#action-buttons) to build, but if you're unable due to some reason, you can use the commands below

  ```bash
  colcon build --symlink-install --packages-up-to robot_bringup
  source install/setup.bash
  ```

  Note: Ensure you've run `Import Libs` atleast once (or the equivalent command)
- **To setup WiCAN** (only if working on hardware)
    ```bash
    sudo slcand -o -s8 -t sw -S 3000000 /dev/ttyUSB0 can0
    sudo ifconfig can0 txqueuelen 1000
    sudo ifconfig can0 up
    ```
    Note: You need to change `ttyUSB0` to the WiCAN device
- **Launch**

  ```bash
  ros2 launch robot_bringup robot_bringup.launch.py use_sim_time:=False
  ```
  Available arguments are as follows:

  | Argument      | Default Value | Description                                    |
  |--------------|--------------|------------------------------------------------|
  | `use_sim_time` | `False`      | Launch in simulation mode (`True` for simulation, `False` for hardware connection). |
  | `namespace`   | `""` (empty)  | Namespace for the launched nodes.              |
  | `record`      | `False`      | Enable recording to a rosbag.                  |
  | `use_rviz`    | `False`      | Launch RViz on startup.                        |
  | `use_joy`    | `False`      | Use joystick control.                        |

- **Interact**

  To interact with the drive, you can use the `teleop_twist_keyboard` node by launching it as follows
  ```bash
  ros2 run teleop_twist_keyboard teleop_twist_keyboard
  ```
  or make the arg `use_joy:=True` while launching when you have a Joystick Controller connected


- **SLAM-toolbox**
```bash
  ros2 launch robot_bringup online_async_launch.py
```

- **NAV2**
  ```bash
  ros2 launch robot_bringup navigation_launch.py use_sim_time:=true
  ```


Note: The README's in this repository are inspired by [this](https://github.com/TheProjectsGuy/MR21-CS7.503)