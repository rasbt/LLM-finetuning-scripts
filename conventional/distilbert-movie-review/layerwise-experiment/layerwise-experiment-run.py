#!/usr/bin/env python
import subprocess

with open("layerwise-experiment-results.txt", "w+") as output:
    subprocess.call(["python", "layerwise-experiment.py"], stdout=output);

########

s = ("Sanity", "Testing", "Validation", "Epoch")

out = []
with open("layerwise-experiment-results.txt", "r") as f:
    for line in f.readlines():
        if line.startswith(s) or not line.strip():
            continue

        else:
            out.append(line)
with open("layerwise-experiment-results-clean.txt", "w") as f:
    for line in out:
        f.write(line)