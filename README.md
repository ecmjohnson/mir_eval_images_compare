# Prerequisites

Full installation of MATLAB is required to run the comparison. ```mir_eval_images```, ```mir_eval_images_noperm``` and ```mir_eval_sources``` can still be run without MATLAB being installed.

# Installation

Ensure MATLAB is installed and can be found by matlab-wrapper. Ensure a version of mir_eval with images support is installed. Testing data is distributed using git lfs. Check the files have downloaded correctly. Run ```pip install -r requirements.txt```

# Example test

```python
python compare.py data/Targets/ANiMAL\ -\ Rockshow/ data/Estimates/ANiMAL\ -\ Rockshow/
```
