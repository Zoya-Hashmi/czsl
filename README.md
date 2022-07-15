
# Compositional Zero-Shot Learning

Helpful Commands
### Creating Python Environment
```
    conda env create --file environment.yml
    conda activate czsl
```
### Downloading Data
```
    bash ./utils/download_data.sh DATA_ROOT
    mkdir logs
```
### Training
```
    python train.py --config CONFIG_FILE
    python train.py --config configs/conditional/mit.yml
```
where `CONFIG_FILE` is the path to the configuration file of the model. 
### Testing
```
    python test.py --logpath LOG_DIR
```
where `LOG_DIR` is the directory containing the logs of a model.
