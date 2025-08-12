
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ASRTransformer
from data_loader import ASRDataset, collate_fn
from tokenizer import CharTokenizer

def train_model(model, data_loader, optimizer, criterion, tokenizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (mels, mfccs, transcripts, mel_lengths, transcript_lengths) in enumerate(data_loader):
            mels = mels.to(device)
            transcripts = transcripts.to(device)

            decoder_input = torch.cat([torch.zeros(transcripts.shape[0], 1, dtype=torch.long, device=device), transcripts[:, :-1]], dim=1)
            
            output = model(mels, decoder_input)

            output = output.contiguous().view(-1, output.shape[-1])
            target = transcripts.contiguous().view(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished, Average Loss: {total_loss / len(data_loader):.4f}")

if __name__ == "__main__":
    # Hyperparameters
    n_feature = 80  # Mel-spectrogram features
    n_head = 4 # Reduced from 8
    n_hid = 128 # Reduced from 256
    n_encoder_layers = 2 # Reduced from 3
    n_decoder_layers = 2 # Reduced from 3
    dropout = 0.1
    batch_size = 2 # Reduced from 4
    learning_rate = 0.001
    epochs = 1

    data_dir = "./data"
    chars = "abcdefghijklmnopqrstuvwxyz \n'"
    tokenizer = CharTokenizer(chars)
    n_class = tokenizer.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASRTransformer(n_feature, n_head, n_hid, n_encoder_layers, n_decoder_layers, n_class, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding index 0

    dataset = ASRDataset(data_dir, subset="dev-clean", tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print("Starting training...")
    train_model(model, data_loader, optimizer, criterion, tokenizer, device, epochs)
    print("Training finished.")

    # Save the trained model
    torch.save(model.state_dict(), "asr_model.pth")
    print("Model saved to asr_model.pth")


