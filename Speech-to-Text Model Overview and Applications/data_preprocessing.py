
import torchaudio
import torchaudio.transforms as T
import torch

def process_audio(waveform, sample_rate=16000, n_mels=80, n_mfcc=40):
    # If the input is a tuple (waveform, sr), extract them
    if isinstance(waveform, tuple):
        waveform, sr = waveform
    else:
        # Assume waveform is already a tensor and sample_rate is provided
        sr = sample_rate

    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Ensure waveform is 2D (batch, samples) for MelSpectrogram
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0) # Add a batch dimension if missing

    # Mel-spectrogram
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # MFCC
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={
        "n_mels": n_mels
    })
    mfcc = mfcc_transform(waveform)

    return mel_spectrogram, mfcc

if __name__ == "__main__":
    # Example usage with a dummy waveform
    print("Data preprocessing script created. You can use the process_audio function to extract features.")
    print("Next, we need to integrate this with a data loading pipeline for the dataset.")
    
    # Create a dummy waveform for testing
    dummy_waveform = torch.randn(1, 16000) # 1 second of mono audio at 16kHz
    dummy_sample_rate = 16000
    
    mel, mfcc = process_audio(dummy_waveform, dummy_sample_rate)
    print(f"Dummy Mel-spectrogram shape: {mel.shape}")
    print(f"Dummy MFCC shape: {mfcc.shape}")


