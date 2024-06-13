## AIC2024

The default data path is hardcoded as data/aic24/test/scene_{scene_number}
If used in SCU WAVE HPC, this can be conducted by using the `make_link.sh` to create symbolic link from the /WAVE/dataset folder

## Environments and reproduction

# for detection
the detection environment is exported in `detection_requirements.txt`

the script for unning the detection is named `rep_detection.sh`, default scene number is set as scene_061.

# for re-id
The re-id environment is exported in `reid_requirements.txt`

the script for running the re-id is named `rep_feature_extraction.sh`

# for single-camera-tracking
the sct environment is exported as `sct_requirements.txt`

the script for running the sct is named `rep_sct_track.sh`

# for feature-filtering, clustering, reassignment
The sct environment should be sufficient for these tasks

For these steps, please run `rep_clustering`, `rep_stcra` in order. 


