#!/usr/bin/env bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt-get update -q
sudo apt-get install -q ros-melodic-ros-base ros-melodic-tf2-geometry-msgs
sudo rosdep init -q
rosdep update -q
