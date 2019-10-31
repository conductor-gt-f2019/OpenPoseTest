"""Starts OpenPose and creates an opWrapper object"""
import sys
import os
import json

# Import OpenPose paths from JSON
OP_PATH = {}
OP_PATH = json.load(open("opPath.json"))

# Add OpenPose compiled directories to path
sys.path.append(OP_PATH["pyRelease"])
os.environ['PATH'] = os.environ['PATH'] + ";" + OP_PATH["x64Release"] + ";" + OP_PATH["bin"] + ";"
import pyopenpose as op  # pylint: disable=import-error, wrong-import-position

# Load OpenPose defaulat parameters
params = dict()
params["model_folder"] = OP_PATH["models"]

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
