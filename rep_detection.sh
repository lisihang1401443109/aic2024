#!/bin/bash


# detection
cd yolov8
scene="scene_061"
python test_predictor.py -s "$scene" 
