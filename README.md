# SpinTRAK

## Environment set up

### Prerequisites
* Python version `3.12`
* CUDA 12.8

### Set up
To set up the environment run the following commands:
* `uv venv --python 3.12`
* `source venv/bin/activate`
* `uv pip install -e .`

## Running API

 * Before running the API, generate Gradient for the entire dataset (phase1)
 
`spintrak generate-training-dataset-parallel \
    --input /home/ubuntu/phase1_data \
    --model /home/ubuntu/musicgen_finetunes/Phase1-Checkpoint \
    --output /home/ubuntu/phase1_gradients.bin \
    --processes-per-gpu 1`

* From the API-fold-artist-spin-trak\src\spintrak folder run the below commands to run the API

`uvicorn influence_api:app --host 0.0.0.0 --port 8002 --reload`
