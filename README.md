# individualized_calibration_for_language
individualized calibration for encoder language models based on https://arxiv.org/abs/2006.10288

## Requirements
- pytorch 2.0 or higher with cuda 11.8
- transformers 4.0 or higher
- numpy
- pandas
- sklearn
- tqdm
- matplotlib
- accelerate
- datasets

## Usage
### 1. Download the data from Kaggle
### 2. Set up the accelerated training configuration
`accelerate config`
### 3. Run the training script

The primary training script to run is `main2.py`. 

The script can be run with the following command:

`accelerate launch main2.py`

Note the flags that can be set in the script, which are described in `flags.py`.
