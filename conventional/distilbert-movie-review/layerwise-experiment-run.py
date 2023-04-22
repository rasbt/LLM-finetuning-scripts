#!/usr/bin/env python
import subprocess

with open("layerwise-experiment-results.txt", "w+") as output:
        subprocess.call(["python", "layerwise-experiment.py"], stdout=output);
