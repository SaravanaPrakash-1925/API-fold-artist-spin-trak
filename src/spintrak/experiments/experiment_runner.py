import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch
import typer
import yaml
import multiprocessing as mp

from spintrak.gradients import worker
from spintrak.utils import setup_logging

setup_logging()
app = typer.Typer()


@app.command()
def generate_output_with_durations(
    config_path: str = typer.Option(..., "--config", help="Path to experiment config"),
    resume_from: str = typer.Option("", help=""),
    processes_per_gpu: int = typer.Option(2, help="Number of processes per GPU"),
):
    with open(f"{config_path}", "r") as f:
        config = yaml.safe_load(f)
        
    experiment = config.get("experiment", {})
    input_section = config.get("input", {})

    experiment_name = experiment.get("name", "unknown_experiment")
    models = experiment.get("models", [])
    cuda_number = experiment.get("gpu_id") or 0

    prompts = input_section.get("prompts", [])
    seeds = input_section.get("seeds", [])
    durations = input_section.get("durations", [])

    model_path = models[0] if models else None
    device_id = f"cuda:{cuda_number}"
    script_dir = Path(os.getcwd()).resolve()
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_embeddings = manager.list()
    num_gpus = torch.cuda.device_count()
    gpu_semaphores = {
        gpu_id: manager.BoundedSemaphore(processes_per_gpu)
        for gpu_id in range(num_gpus)
    }

    if resume_from:
        experiment_dir = Path(resume_from)
        assert experiment_dir.exists() and experiment_dir.is_dir()
        all_generations = [
            torch.load(partial_result) for partial_result in experiment_dir.glob("*.pt")
        ]
        for gen in all_generations:
            shared_embeddings.append(gen)
    else:
        ts = int(time.time())
        experiment_dir = script_dir / f"{experiment_name}_{ts}"
        experiment_dir.mkdir(exist_ok=True)
    with open( experiment_dir / f'config_{experiment_name}.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    configs = []
    for prompt in prompts:
        for seed in seeds:
            for duration in durations:
                name = f"{'_'.join(prompt.strip().split()[:3])}_{duration}_{seed}"
                full_name = experiment_dir / name
                if resume_from and full_name.with_suffix(".pt").exists():
                    logging.info(
                        f"Skipping already calculated test case: {full_name.stem}"
                    )
                    continue
                configs.append(
                    {
                        "name": name,
                        "seed": seed,
                        "prompt": prompt,
                        "duration": duration,
                        "directory": experiment_dir,
                    }
                )
    task_queue = mp.Queue()
    for file in configs:
        task_queue.put(file)

    gpu_ids = []
    for gpu in range(num_gpus):
        gpu_ids.extend([gpu] * processes_per_gpu)

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        semaphore = gpu_semaphores[gpu_id]
        p = mp.Process(
            target=worker,
            args=(
                i,
                gpu_id,
                task_queue,
                semaphore,
                model_path,
                experiment_dir,
                duration,
                shared_embeddings,
                0,
            ),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    if len(shared_embeddings) > 0:
        torch.save(
            list(shared_embeddings), experiment_dir / f"output_{experiment_name}.pt"
        )
        logging.info(f"{len(shared_embeddings)} gradients saved")
        logging.info(
            f"For given seeds and durations, {len(shared_embeddings)} outputs generated"
        )


if __name__ == "__main__":
    app()
