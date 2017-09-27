# PNNL-Image-Hackathon
A git repo to keep track of PNNL LIDC hackathon

# Notes

## When a frame buffer is unavailable

When running in environments without a display (e.g. on a cluster or remote server without X), the pylidc library will error and dump core:

```
In [1]: QXcbConnection: Could not connect to display 
Aborted (core dumped)
```

Install Xvfb and either run Xvfb as a daemon or use `xvfb-run` which will start the frame buffer and run the process within that environment:

    xvfb-run pylidc-script.py


