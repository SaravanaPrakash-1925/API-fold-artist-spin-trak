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

## Running single generation
`spintrak generate-audio --prompt "A calm piano melody" --duration 30`

## Running single attribution scores calculation
`spintrak generate-gradients --prompt "calm ocean waves" --duration 30 --output ./calm_ocean_gradients.pt`

## Running experiments

To run experiments please refer to the [Experiments.md](./Experiments.md) file in the `experiments` directory.
