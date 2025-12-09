import gc
import json
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import pynvml
import torch
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from trak.projectors import CudaProjector, ProjectionType

from fold_audiocraft.data.audio import audio_write
from fold_audiocraft.models import MusicGen
from spintrak.initialization import DEFAULT_SEED, setup_deterministic_environment
from spintrak.modified_audiocraft_generation import custom_generate
from spintrak.spin_trak import SPINTRAK
from spintrak.utils import LOCAL_MODEL_DIR


def get_gradient_embeddings(
    model, prompt=None, audio_input=None, output_duration=4, gpu_id=0
):
    model.set_generation_params(duration=output_duration)
    model.compression_model.train()
    model.lm.requires_grad_()
    model.compression_model.requires_grad_()
    device = model.device

    logging.info("Starting generation")
    start = time.perf_counter()
    outputs, generated_tokens = custom_generate(
        model,
        descriptions=[prompt],
        progress=True,
        return_tokens=True,
        save_grads=True,
    )
    end = time.perf_counter()
    logging.info(f"Generation time elapsed: {end - start:.2f}s")

    logging.info("Gathering gradients")
    f = outputs.mean()
    model.lm.reset_streaming()
    model.lm.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    f.backward()
    decoder_params = (
        model.compression_model.decoder.parameters()
        if hasattr(model.compression_model, "decoder")
        else model.compression_model.model.model.decoder.parameters()
    )
    decoder_grad_list = [param.grad for param in decoder_params if param.requires_grad]
    with torch.no_grad():
        wait_for_free_memory(gpu_id, required_gb=15.0)
        gradients_decoder = torch.cat([x.flatten().cpu() for x in decoder_grad_list])
        ###
        wait_for_free_memory(gpu_id, required_gb=15.0)
        gradients_lm = (
            torch.cat([x.flatten().cpu() for x in model.lm.gradients])
            / generated_tokens.shape[2]
        )
        all_gradients = torch.cat([gradients_decoder, gradients_lm])
        all_gradients = all_gradients.to(device)
        del gradients_decoder
        del gradients_lm
        del decoder_grad_list
        model.lm.gradients = None
        logging.debug("Starting projections")
        model_size = all_gradients.shape[0]
        projected_dim = johnson_lindenstrauss_min_dim(all_gradients.shape[0], eps=0.15)
        projected_dim = (projected_dim // 512) * 512
        logging.info(f"Projecting gradients from {model_size} to {projected_dim}")
        projector = CudaProjector(
            grad_dim=model_size,
            proj_dim=projected_dim,
            seed=42,
            proj_type=ProjectionType.normal,
            device=device,
            max_batch_size=16,
        )
        all_gradients = all_gradients.to(device)
        projected = projector.project(all_gradients.unsqueeze(0), model_id=0)
        logging.debug("Projection ended")
        return outputs, projected


def create_training_dataset(model, input_path, output_path, resume=True):
    directory = Path(input_path)
    assert directory.exists() and directory.is_dir(), f"input directory corrupted"
    music_files = []
    embeddings_list = []
    music_dirs = set()
    music_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
    descriptions = [f.stem for f in directory.rglob("*.json")]
    existing_gradients = [f.stem for f in directory.rglob("*.pt")]
    for file in directory.rglob("*"):
        if (
            file.is_file()
            and file.suffix.lower() in music_extensions
            and file.stem in descriptions
            and (
                not resume
                or not (
                    file.parent / "gradients" / file.with_suffix(".pt").name
                ).exists()
            )
        ):
            music_files.append(file)
            music_dirs.add(file.parent)

    for music_directory in music_dirs:
        generated_path = music_directory / "gradients"
        generated_path.mkdir(exist_ok=True)

    for music_file in music_files:
        json_file = music_file.with_suffix(".json")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt = data["description"]
            duration = data["duration"]
        logging.info(f"Getting gradients from {music_file}")
        _, embeddings = get_gradient_embeddings(
            model, prompt=prompt, output_duration=duration
        )
        single_gradients_file = (
            music_file.parent / "gradients"
        ) / music_file.with_suffix(".pt").name
        torch.save(embeddings, single_gradients_file)

        embeddings_list.append(embeddings)
    if embeddings_list:
        dataset = torch.cat(embeddings_list, dim=0)
        torch.save(dataset, output_path)
        logging.info(f"{len(embeddings_list)} gradients saved")
    else:
        logging.info("0 gradients saved")


def file_to_item(path: Path, duration, seed):
    json_file = path.with_suffix(".json")
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        prompt = data["description"]
        duration = data["duration"] if duration is None else duration
    return {
        "prompt": prompt,
        "duration": duration,
        "name": path.name,
        "seed": seed,
    }


def worker(
    process_id,
    gpu_id,
    task_queue,
    semaphore,
    model_path,
    output_path,
    duration,
    embeddings_list,
    files=True,
    seed=None,
):
    torch.cuda.set_device(gpu_id)
    with semaphore:
        try:
            while True:
                try:
                    item = task_queue.get_nowait()
                except mp.queues.Empty:
                    break
                if model_path is not None:
                    model = MusicGen.get_pretrained(Path(model_path))
                else:
                    model = MusicGen.get_pretrained(Path(LOCAL_MODEL_DIR) / "fb-large/")
                if files:
                    single_gradients_file = (
                        item.parent / "gradients"
                    ) / item.with_suffix(".pt").name
                    item = file_to_item(item, duration, seed)
                name = item["name"]
                prompt = item["prompt"]
                duration = item["duration"]
                seed = item["seed"]
                setup_deterministic_environment(seed)

                logging.info(f"Getting gradients from {item}")
                wav, embeddings = get_gradient_embeddings(
                    model, prompt=prompt, output_duration=duration, gpu_id=gpu_id
                )
                embeddings = embeddings.detach().cpu()
                item["gradients"] = embeddings

                embeddings_list.append(item)
                if files:
                    torch.save(embeddings, single_gradients_file)
                else:
                    logging.info("Saving audio files")
                    torch.save(item, f"{output_path}/{name}.pt")
                    for one_wav in wav:
                        audio_write(
                            f"{output_path}/{name}",
                            one_wav.cpu(),
                            model.sample_rate,
                            strategy="loudness",
                        )
                del model
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e:
            logging.exception(f"[Process {process_id}] Crashed", exc_info=e)


def wait_for_free_memory(
    gpu_id: int, required_gb: float, check_interval: float = 3.0, max_retries: int = 60
):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    for _ in range(max_retries):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = mem_info.free / 1024**3
        if free_gb >= required_gb:
            return True
        time.sleep(check_interval)
        print("wait for free card")

    raise RuntimeError(
        f"Timeout: GPU {gpu_id} did not free up {required_gb} GB within {max_retries * check_interval}s"
    )


def create_training_dataset_parallel(
    input_path, output_path, seed, resume=True, model_path=None, duration=None,
    processes_per_gpu: int = 2,
):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_embeddings = manager.list()
    num_gpus = torch.cuda.device_count()
    gpu_semaphores = {
        gpu_id: manager.BoundedSemaphore(processes_per_gpu)
        for gpu_id in range(num_gpus)
    }

    directory = Path(input_path)
    assert directory.exists() and directory.is_dir(), "Input directory corrupted"

    music_files = []
    music_dirs = set()
    music_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
    descriptions = [f.stem for f in directory.rglob("*.json")]

    for file in directory.rglob("*"):
        if (
            file.is_file()
            and file.suffix.lower() in music_extensions
            and file.stem in descriptions
            and (
                not resume
                or not (
                    file.parent / "gradients" / file.with_suffix(".pt").name
                ).exists()
            )
        ):
            music_files.append(file)
            music_dirs.add(file.parent)

    for music_directory in music_dirs:
        (music_directory / "gradients").mkdir(exist_ok=True)

    task_queue = mp.Queue()
    for file in music_files:
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
                output_path,
                duration,
                shared_embeddings,
                1,
                seed
            ),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    if len(shared_embeddings) > 0:
        torch.save(list(shared_embeddings), output_path)
        logging.info(f"{len(shared_embeddings)} gradients saved")
