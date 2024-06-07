#!/bin/bash


# reid feature extraction
cd UCMCTrack
scene="scene_061"
python aic_demo.py --scene "$scene" --a 15 --high_score 0.8 
