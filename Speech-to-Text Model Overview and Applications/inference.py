import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

class ASRInference:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
        Initialize the ASR inference system with a pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio from a file path.
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to numpy and squeeze
        audio_input = waveform.squeeze().numpy()
        
        # Process audio
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def transcribe_waveform(self, waveform, sample_rate=16000):
        """
        Transcribe audio from a waveform tensor.
        """
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to numpy and squeeze
        audio_input = waveform.squeeze().numpy()
        
        # Process audio
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription

if __name__ == "__main__":
    # Test the inference system
    asr = ASRInference()
    
    # Create a dummy audio file for testing
    dummy_waveform = torch.randn(1, 16000)  # 1 second of audio
    transcription = asr.transcribe_waveform(dummy_waveform)
    
    print(f"Dummy transcription: {transcription}")
    print("ASR inference system ready!")

