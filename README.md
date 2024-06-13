## AIC2024

The default data path is hardcoded as data/aic24/test/scene_{scene_number}
If used in SCU WAVE HPC, this can be conducted by using the `make_link.sh` to create symbolic link from the /WAVE/dataset folder

## Environments and reproduction

# for detection
the detection environment is exported in `detection_requirements.txt`

the script for running the detection is named `rep_detection.sh`, default scene number is set as scene_061.

# for re-id
The re-id environment is exported in `reid_requirements.txt`

the script for running the re-id is named `rep_feature_extraction.sh`

# for single-camera-tracking
the sct environment is exported as `sct_requirements.txt`

the script for running the sct is named `rep_sct_track.sh`

# for feature-filtering, clustering, reassignment
The sct environment should be sufficient for these tasks

For these steps, please run `rep_clustering`, `rep_stcra` in order. 

## reproduce pipeline

the scripts starting with `rep_{}.sh` are designed to reproduce the result on a single scene in the testing folder, (defalt is scene_061). Scripts are designed to be run under the root folder of this project as they will cd into their respective folders. To reproduce our result, please run the scripts in the root in the following order:

```bash

# change to the detection environment
conda activate {your_detection_env}

# run the detection

bash rep_detection.sh

# change to the re-id environment
conda activate {your_reid_env}

# run the re-id
bash rep_feature_extraction.sh

# change to the single-camera-tracking environment
conda activate {your_sct_env}

# run the single camera tracking
bash rep_sct_track.sh

# run the filtering and clustering
bash rep_clustering.sh

# run the stcra (optional)
bash rep_stcra.sh

# prepare for the AIC_format (optional)

bash rep_submission.sh

```


