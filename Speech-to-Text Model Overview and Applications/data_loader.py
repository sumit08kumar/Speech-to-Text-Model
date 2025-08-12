
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import os
from data_preprocessing import process_audio
from tokenizer import CharTokenizer

class ASRDataset(Dataset):
    def __init__(self, root_dir, subset="dev-clean", sample_rate=16000, n_mels=80, n_mfcc=40, tokenizer=None):
        self.librispeech = torchaudio.datasets.LIBRISPEECH(root_dir, url=subset, download=True)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.librispeech)

    def __getitem__(self, idx):
        waveform, sr, transcript, speaker_id, chapter_id, utterance_id = self.librispeech[idx]

        mel_spectrogram, mfcc = process_audio(waveform, sample_rate=sr, n_mels=self.n_mels, n_mfcc=self.n_mfcc)

        if self.tokenizer:
            transcript_tokens = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)
        else:
            transcript_tokens = transcript

        # Transpose mel_spectrogram and mfcc to be (features, sequence_length)
        # The model expects (batch_size, sequence_length, features)
        return mel_spectrogram.squeeze(0).transpose(0, 1), mfcc.squeeze(0).transpose(0, 1), transcript_tokens

def collate_fn(batch):
    # Pad the sequences to the maximum length in the batch
    mel_spectrograms, mfccs, transcripts = zip(*batch)

    # Find max lengths
    max_mel_len = max(mel.shape[0] for mel in mel_spectrograms) # Now features are dim 1, sequence length is dim 0
    max_mfcc_len = max(mfcc.shape[0] for mfcc in mfccs)
    max_transcript_len = max(len(t) for t in transcripts)

    padded_mel_spectrograms = []
    padded_mfccs = []
    padded_transcripts = []
    mel_lengths = []
    transcript_lengths = []

    for mel, mfcc, transcript in batch:
        # Pad Mel-spectrograms
        pad_mel = max_mel_len - mel.shape[0]
        padded_mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_mel)) # Pad along sequence length dimension
        padded_mel_spectrograms.append(padded_mel)
        mel_lengths.append(mel.shape[0])

        # Pad MFCCs (if needed, though not used in Transformer directly)
        pad_mfcc = max_mfcc_len - mfcc.shape[0]
        padded_mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, pad_mfcc))
        padded_mfccs.append(padded_mfcc)

        # Pad transcripts
        pad_transcript = max_transcript_len - len(transcript)
        padded_transcript = torch.nn.functional.pad(transcript, (0, pad_transcript), "constant", 0) # Pad with 0 (blank)
        padded_transcripts.append(padded_transcript)
        transcript_lengths.append(len(transcript))

    return (
        torch.stack(padded_mel_spectrograms),
        torch.stack(padded_mfccs),
        torch.stack(padded_transcripts),
        torch.tensor(mel_lengths, dtype=torch.long),
        torch.tensor(transcript_lengths, dtype=torch.long)
    )

if __name__ == "__main__":
    data_dir = "./data"
    
    # Define a simple character set for the tokenizer
    chars = "abcdefghijklmnopqrstuvwxyz \n'"
    tokenizer = CharTokenizer(chars)

    dataset = ASRDataset(data_dir, subset="dev-clean", tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"Number of samples in LibriSpeech dev-clean: {len(dataset)}")

    for i, (mels, mfccs, transcripts, mel_lengths, transcript_lengths) in enumerate(data_loader):
        print(f"Batch {i+1}:")
        print(f"  Mel-spectrograms shape: {mels.shape}")
        print(f"  MFCCs shape: {mfccs.shape}")
        print(f"  Transcripts shape: {transcripts.shape}")
        print(f"  Mel lengths: {mel_lengths}")
        print(f"  Transcript lengths: {transcript_lengths}")
        print(f"  Decoded transcript (first item): {tokenizer.decode(transcripts[0].tolist())}")
        if i == 0: # Print only the first batch for brevity
            break


