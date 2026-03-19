# Pretrained Checkpoints

This directory holds pretrained GazeMAE models used by the TrustME embedding pipeline.

## Included Models

- `pos-i3738`: position-signal autoencoder checkpoint (iteration 3738)
- `vel-i8528`: velocity-signal autoencoder checkpoint (iteration 8528)

Both checkpoints are hierarchical models with effective 128-d latent output (`64 x 2`).

## Used By

- `trustME/build_gazemae_embeddings.py` via:
  `--model-pos models/pos-i3738 --model-vel models/vel-i8528`
