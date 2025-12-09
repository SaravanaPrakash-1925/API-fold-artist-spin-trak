# Experiments

Each experiment is described by it's config in the `yaml` format.

To reproduce an experiment follow these steps:

1. Set up and activate an environment, as described in the [README.md](./README.md) file.
2. Create training dataset gradient embeddings by running:
    ```bash
    spintrak generate-training-dataset-parallel --input <path to the training dataset directory> --model <path to the directory with compressed_state_dict.bin and state_dict.bin files> --duration 30 --output training_gradients_30.bin
    ```
    You can change the `duration` value to suit your needs. Note that you might need to run this command multiple times in order to calculate multiple training dataset embeddings for different durations, for example experiment 1 requires 3 different durations.

3. Run a specific experiment, making sure that the duration(s) specified in the config matches the ones computed in step 1. To run the experiment run:
    ```bash
    experiment_runner --config <path to experiment config yaml file>
    ```
    This command will generate a new directory, which name will consist of the name in the config and the timestamp. All of the calculated gradient embeddings and generated audio will be stored in this directory.
4. Create results visualization using:
    ```bash
    experiment_X_results <path to the output .pt file> <path(s) to the training gradients file>
    ```

In the following sections we show examples of each experiment being run, note that your paths may differ. You can also run the last step to recalculate the results based on the files provided in the [Google Drive](https://drive.google.com/drive/folders/1Si6E7aDu9RbvTPw6bMwp-h-HLZtLHneK) with our results.

## Experiment 1
This experiment requires different durations for the training dataset embeddings.

```bash
# 1. Calculate training dataset embeddings
spintrak generate-training-dataset-parallel --input /mnt/remote-storage/new_finetunings/two_songs/dataset/ --model /mnt/remote-storage/new_finetunings/two_songs/weights/ --duration 30 --output two_songs_training_gradients_30.bin
spintrak generate-training-dataset-parallel --input /mnt/remote-storage/new_finetunings/two_songs/dataset/ --model /mnt/remote-storage/new_finetunings/two_songs/weights/ --duration 60 --output two_songs_training_gradients_60.bin
spintrak generate-training-dataset-parallel --input /mnt/remote-storage/new_finetunings/two_songs/dataset/ --model /mnt/remote-storage/new_finetunings/two_songs/weights/ --duration 120 --output two_songs_training_gradients_120.bin
# 2. Calculate embeddings for given experimental prompts, seeds and durations
experiment_runner --config ./experiments_configs/experiment_1.yaml
# 3. Generate experimental results
experiment_1_results ./experiment_1_1758810259/output_experiment_1.pt ./two_songs_training_gradients_*.bin
```

## Experiment 3

```bash
# 1. Calculate training dataset embeddings
# You can skip this step if you already generated two_songs_training_gradients_30.bin
spintrak generate-training-dataset-parallel --input /mnt/remote-storage/new_finetunings/two_songs_temp/dataset/ --model /mnt/remote-storage/new_finetunings/two_songs/weights/ --duration 30 --output two_songs_training_gradients_30.bin
# 2. Calculate embeddings for given experimental prompts, seeds and durations
experiment_runner --config ./experiments_configs/experiment_3.yaml
# 3. Generate experimental results
experiment_3_results ./experiment_3_1758808575/output_experiment_3.pt ./two_songs_training_gradients_30.bin
```

## Experiment 4

```bash
# 1. Calculate training dataset embeddings
# You can skip this step if you already generated two_songs_training_gradients_30.bin
spintrak generate-training-dataset-parallel --input /mnt/remote-storage/new_finetunings/two_songs_temp/dataset/ --model /mnt/remote-storage/new_finetunings/two_songs/weights/ --duration 30 --output two_songs_training_gradients_30.bin
# 2. Calculate embeddings for given experimental prompts, seeds and durations
experiment_runner --config ./experiments_configs/experiment_4.yaml
# 3. Generate experimental results
experiment_4_results ./experiment_4_1758882717/output_experiment_4.pt ./two_songs_training_gradients_30.bin
```
