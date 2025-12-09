import io
import torchaudio
import typer


def generate_with_musicgen(model, prompt=None, audio_input=None, output_duration=8):
    model.set_generation_params(duration=output_duration)
    audio_tensor = None
    if audio_input is not None:
        with io.BytesIO(audio_input) as f:
            waveform, sr = torchaudio.load(f)
        if sr != model.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                waveform, sr, model.sample_rate
            )
        else:
            audio_tensor = waveform
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.device != model.device:
            audio_tensor = audio_tensor.to(model.device)

    if prompt and audio_tensor is not None:
        print(f"tokens: {prompt}")
        wav = model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=[audio_tensor],
            melody_sample_rate=model.sample_rate,
            progress=True,
        )
    elif prompt:
        print(f"tokens: {prompt}")
        wav = model.generate(
            descriptions=[prompt],
            progress=True,
        )
    elif audio_tensor is not None:
        wav = model.generate_with_chroma(
            descriptions=[""],
            melody_wavs=[audio_tensor],
            melody_sample_rate=model.sample_rate,
            progress=True,
        )
    return wav

