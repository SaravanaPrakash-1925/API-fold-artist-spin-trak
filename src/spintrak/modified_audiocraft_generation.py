import importlib
import sys
import typing as tp
import torch
from fold_audiocraft.data.audio_utils import convert_audio
from fold_audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
fold_audiocraft = importlib.import_module("fold_audiocraft")
sys.modules["audiocraft"] = fold_audiocraft


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

def custom_generate(
    model,
    descriptions: tp.List[str],
    progress: bool = False,
    return_tokens: bool = False,
    save_grads: bool = False,
) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    """Generate samples conditioned on text.

    Args:
        descriptions (list of str): A list of strings used as text conditioning.
        progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    """
    attributes, prompt_tokens = custom_prepare_tokens_and_attributes(
        model, descriptions, None
    )
    assert prompt_tokens is None
    tokens = custom_generate_tokens(
        model, attributes, prompt_tokens, progress, save_grads
    )
    if return_tokens:
        return custom_generate_audio(model, tokens), tokens
    return custom_generate_audio(model, tokens)


def custom_generate_tokens(
    model,
    attributes: tp.List[ConditioningAttributes],
    prompt_tokens: tp.Optional[torch.Tensor],
    progress: bool = False,
    save_grads: bool = False,
) -> torch.Tensor:
    total_gen_len = int(model.duration * model.frame_rate)
    max_prompt_len = int(min(model.duration, model.max_duration) * model.frame_rate)
    current_gen_offset: int = 0

    def _progress_callback(generated_tokens: int, tokens_to_generate: int):
        generated_tokens += current_gen_offset
        if model._progress_callback is not None:
            # Note that total_gen_len might be quite wrong depending on the
            # codebook pattern used, but with delay it is almost accurate.
            model._progress_callback(generated_tokens, tokens_to_generate)
        else:
            print(f"{generated_tokens: 6d} / {tokens_to_generate: 6d}", end="\r")

    if prompt_tokens is not None:
        assert max_prompt_len >= prompt_tokens.shape[-1], (
            "Prompt is longer than audio to generate"
        )

    callback = None
    if progress:
        callback = _progress_callback

    if model.duration <= model.max_duration:
        # generate by sampling from LM, simple case.
        with model.autocast:
            gen_tokens = model.lm.generate(
                prompt_tokens,
                attributes,
                callback=callback,
                max_gen_len=total_gen_len,
                save_grads=save_grads,
                **model.generation_params,
            )
    else:
        # now this gets a bit messier, we need to handle prompts,
        # melody conditioning etc.
        ref_wavs = [attr.wav['self_wav'] for attr in attributes]
        all_tokens = []
        if prompt_tokens is None:
            prompt_length = 0
        else:
            all_tokens.append(prompt_tokens)
            prompt_length = prompt_tokens.shape[-1]

        assert model.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
        assert model.extend_stride < model.max_duration, "Cannot stride by more than max generation duration."
        stride_tokens = int(model.frame_rate * model.extend_stride)

        while current_gen_offset + prompt_length < total_gen_len:
            time_offset = current_gen_offset / model.frame_rate
            chunk_duration = min(model.duration - time_offset, model.max_duration)
            max_gen_len = int(chunk_duration * model.frame_rate)
            for attr, ref_wav in zip(attributes, ref_wavs):
                wav_length = ref_wav.length.item()
                if wav_length == 0:
                    continue
                # We will extend the wav periodically if it not long enough.
                # we have to do it here rather than in conditioners.py as otherwise
                # we wouldn't have the full wav.
                initial_position = int(time_offset * model.sample_rate)
                wav_target_length = int(model.max_duration * model.sample_rate)
                positions = torch.arange(initial_position,
                                         initial_position + wav_target_length, device=model.device)
                attr.wav['self_wav'] = WavCondition(
                    ref_wav[0][..., positions % wav_length],
                    torch.full_like(ref_wav[1], wav_target_length),
                    [model.sample_rate] * ref_wav[0].size(0),
                    [None], [0.])
            with model.autocast:
                gen_tokens = model.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=max_gen_len, save_grads=save_grads, **model.generation_params)
            if prompt_tokens is None:
                all_tokens.append(gen_tokens)
            else:
                all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
            prompt_tokens = gen_tokens[:, :, stride_tokens:]
            prompt_length = prompt_tokens.shape[-1]
            current_gen_offset += stride_tokens

        gen_tokens = torch.cat(all_tokens, dim=-1)
    return gen_tokens


def custom_generate_audio(model, gen_tokens: torch.Tensor) -> torch.Tensor:
    """Generate Audio from tokens."""
    assert gen_tokens.dim() == 3
    gen_audio = model.compression_model.decode(gen_tokens, None)
    return gen_audio


def custom_prepare_tokens_and_attributes(
        model,
        descriptions: tp.Sequence[tp.Optional[str]],
        prompt: tp.Optional[torch.Tensor],
        melody_wavs: tp.Optional[MelodyList] = None,
) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
    attributes = [
        ConditioningAttributes(text={'description': description})
        for description in descriptions]

    if melody_wavs is None:
        for attr in attributes:
            attr.wav['self_wav'] = WavCondition(
                torch.zeros((1, 1, 1), device=model.device),
                torch.tensor([0], device=model.device),
                sample_rate=[model.sample_rate],
                path=[None])
    else:
        if 'self_wav' not in model.lm.condition_provider.conditioners:
            raise RuntimeError("This model doesn't support melody conditioning. "
                               "Use the `melody` model.")
        assert len(melody_wavs) == len(descriptions), \
            f"number of melody wavs must match number of descriptions! " \
            f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
        for attr, melody in zip(attributes, melody_wavs):
            if melody is None:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=model.device),
                    torch.tensor([0], device=model.device),
                    sample_rate=[model.sample_rate],
                    path=[None])
            else:
                attr.wav['self_wav'] = WavCondition(
                    melody[None].to(device=model.device),
                    torch.tensor([melody.shape[-1]], device=model.device),
                    sample_rate=[model.sample_rate],
                    path=[None],
                )

    if prompt is not None:
        if descriptions is not None:
            assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
        prompt = prompt.to(model.device)
        prompt_tokens, scale = model.compression_model.encode(prompt)
        assert scale is None
    else:
        prompt_tokens = None
    return attributes, prompt_tokens

def custom_generate_with_chroma(
    model,
    descriptions: tp.List[str],
    melody_wavs: MelodyType,
    melody_sample_rate: int,
    progress: bool = False,
    return_tokens: bool = False,
) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(melody_wavs, torch.Tensor):
        if melody_wavs.dim() == 2:
            melody_wavs = melody_wavs[None]
        if melody_wavs.dim() != 3:
            raise ValueError("Melody wavs should have a shape [B, C, T].")
        melody_wavs = list(melody_wavs)
    else:
        for melody in melody_wavs:
            if melody is not None:
                assert melody.dim() == 2, (
                    "One melody in the list has the wrong number of dims."
                )

    melody_wavs = [
        convert_audio(wav, melody_sample_rate, model.sample_rate, model.audio_channels)
        if wav is not None
        else None
        for wav in melody_wavs
    ]
    attributes, prompt_tokens = model._prepare_tokens_and_attributes(
        descriptions=descriptions, prompt=None, melody_wavs=melody_wavs
    )
    assert prompt_tokens is None
    tokens = custom_generate_tokens(model, attributes, prompt_tokens, progress)
    if return_tokens:
        return custom_generate_audio(model, tokens), tokens
    return custom_generate_audio(model, tokens)
