## AIC2024

The default data path is hardcoded as data/aic24/test/scene_{scene_number}
If used in SCU WAVE HPC, this can be conducted by using the `make_link.sh` to create symbolic link from the /WAVE/dataset folder

## Environments and reproduction

# for detection
the detection environment is exported in `detection_requirements.txt` 
```python
conda create --name detection_env python=3.12
conda activate detection_env

#install the dependencies from detection_requirements.txt
pip install -r detection_requirements.txt
```
the script for running the detection is named `rep_detection.sh`, default scene number is set as scene_061.

# for re-id
The re-id environment is exported in `reid_requirements.txt`
```python
conda create --name reid_env python=3.12
conda activate reid_env

#install the dependencies from reid_requirements.txt
pip install -r reid_requirements.txt
```

the script for running the re-id is named `rep_feature_extraction.sh`

# for single-camera-tracking
the sct environment is exported as `sct_requirements.txt` (Note: Please create the conda environment with Python version >3.12.)
```python
conda create --name sct_env python=3.12
conda activate sct_env

#install the dependencies from sct_requirements.txt
pip install -r sct_requirements.txt
```

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

## SCU Senior Design 

Our pipeline was adapted from last year’s winning team’s baseline (
https://github.com/lisihang1401443109/AIC23_Track1_UWIPL_ETRI). We replaced the detector,single_camera_tracking, re_id, with more promising technologies. 

The detector is replaced with Yolov8 by ultralytics. We made modifications to the ultralytic site-packages to change the logging fashions. We also created separate Python script to run the detection by scenario in a steaming fashion.

The single camera tracking was replaced by UCMC track, for which we write a separate detector class that reads the detection files from disk and is compatible with the rest of the code. We also disabled the camera noise and added a class reading holographic matrix from the dataset instead of from the estimator.

The Reid was implemented using fast-Reid library and in specific we used Duke-BoT-s50 model. We adapted the code to work with the AIC dataset, namingly grouping detection by frame, cropping detections from the frame and feed to the model in a batch fashion.

We introduced a part that is not seen in the previous pipeline: the variance preserving filtering, through which we reduced the number of samples without degradation of the clustering performance. 

For the clustering, we did not change much beside changing the distance metric from Euclidean distance to cosine. We also performed some tuning at this stage.

We did not make much modification in the reassignment part, except for some tuning and extracting location information, which is required this year but not last year. We also wrote script that organize the format into aic-submission ready format.
