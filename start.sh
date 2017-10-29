#!/bin/bash

open -a XQuartz
ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $ip

#docker run --name rl -it --rm -p 8888:8888 -p 6006:6006 -v "$(pwd)":/notebooks drl_python:auto

docker run --name drl -it --rm -e DISPLAY=$ip:0 -v /tmp/.X11-unix:/tmp/.X11-uniz -p 8888:8888 -p 6006:6006 -v "$(pwd)":/notebooks drl_python:auto