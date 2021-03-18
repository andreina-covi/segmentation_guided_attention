# Segmentation guided attention
Pytorch implementation for the paper: Reconciling explainability and performance in neural networks by means of semantic segmentation-guided feature attention:An application to urban space perception (unpublished).

## Requirements

Python 3.6.5 and 3.7.1 (only tested on that version)

For more check `requirements.txt`

## Setup and preprocessing

Create a virtual environment (the name in the example was be "venv")

```virtualenv venv -p `which python3.7` ```

Activate the virual environment

`source venv/bin/activate`

Install dependencies

`pip install -r requirements.txt`

Replace two files from torch library instaled in virtual environment:

1. Replace venv/lib/python3.7/site-packages/torch/nn/functional.py with torch-replace/functional.py
2. Replace venv/lib/python3.7/site-packages/torch/nn/modules/activation.py with torch-replace/activation.py

