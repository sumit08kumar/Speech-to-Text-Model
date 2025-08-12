
import torchaudio
from datasets import load_dataset
import os

def download_librispeech(root_dir):
    print("Downloading LibriSpeech (dev-clean, small subset)...")
    # LibriSpeech dataset is large, download only a very small subset for demonstration
    # We will only download the 'dev-clean' split, which is smaller, and then limit further.
    try:
        # Attempt to download a small part of dev-clean
        torchaudio.datasets.LIBRISPEECH(root_dir, url="dev-clean", download=True)
        print("LibriSpeech dev-clean subset downloaded.")
    except Exception as e:
        print(f"Could not download LibriSpeech dev-clean subset: {e}")
        print("Skipping LibriSpeech download due to potential space constraints or network issues.")

def download_tedlium(root_dir):
    print("Skipping TED-LIUM download due to space constraints.")
    pass

def download_common_voice(cache_dir):
    print("Skipping Mozilla Common Voice download due to space constraints.")
    pass

if __name__ == "__main__":
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # Download LibriSpeech (small subset)
    download_librispeech(data_dir)

    # Skipping TED-LIUM and Common Voice for now due to space constraints.
    # We will likely use pre-trained models for these or very small samples if needed later.


