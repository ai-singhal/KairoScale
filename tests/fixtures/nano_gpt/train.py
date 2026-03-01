"""Minimal nanoGPT training script for testing KairoScale.

This is a tiny transformer model with a synthetic dataset,
designed to run in seconds on CPU for integration tests.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NanoGPT(nn.Module):
    """Tiny 2-layer transformer for testing."""

    def __init__(self, vocabSize=256, embedDim=32, nHeads=2, nLayers=2, seqLen=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embedDim)
        self.posEmbedding = nn.Embedding(seqLen, embedDim)
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=embedDim, nhead=nHeads, dim_feedforward=64, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=nLayers)
        self.head = nn.Linear(embedDim, vocabSize)
        self.seqLen = seqLen

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.posEmbedding(positions)
        x = self.transformer(x)
        return self.head(x)


def makeDataset(numSamples=200, seqLen=16, vocabSize=256):
    """Create a synthetic dataset of random token sequences."""
    data = torch.randint(0, vocabSize, (numSamples, seqLen + 1))
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return TensorDataset(inputs, targets)


def train():
    """Standard training loop with loss.backward() + optimizer.step()."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NanoGPT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = makeDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(2):
        for batch in dataloader:
            inputs, targets = [t.to(device) for t in batch]
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(f"Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
