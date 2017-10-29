#!/usr/bin/env bash

#jupyter notebook "$@"

#/usr/bin/xvfb-run -s "-screen 0 1280x720x24" /usr/local/bin/jupyter-notebook --no-browser --ip=0.0.0.0 --notebook-dir=/notebook

/usr/bin/xvfb-run -s "-screen 0 1280x720x24" jupyter notebook "$@"
