# Text-Based Quantile Forecasting with RoBERTa and LSTM Attention

## Overview

I developed this pipeline to forecast firm-level sales from the textual content
of financial disclosures. Rather than predicting a single point estimate, the
framework produces nine simultaneous distributional forecasts (10th through 90th
percentile), capturing uncertainty across the full conditional distribution of
outcomes.

The work combines transformer-based text encoding with a sequence model that learns
to weight the most informative passages within each document, and is applied to
a panel of firm-year observations drawn from financial disclosure filings.

## Key Features

- Sliding-window encoding handles long disclosure texts beyond the 512-token
  transformer limit without hard truncation
- Monotone quantile outputs use softplus-constrained cumulative sums to guarantee
  non-crossing quantile predictions
- Attention-based pooling preserves BiLSTM attention weights for post-hoc inspection
  of which text passages drove each forecast
- Attended-text extraction reconstructs the top-K highest-attention chunks per
  document as readable text saved alongside predictions
- Reproducible caching saves embeddings, sequences, and normalisation statistics
  independently so each stage can be re-run without reprocessing the full dataset

## Technologies

| Category | Tools |
|---|---|
| Language Model | RoBERTa-large (Hugging Face Transformers) |
| Deep Learning | PyTorch, BiLSTM, Attention Pooling |
| Training | AdamW, Pinball Loss, 30 epochs |
| Data I/O | pyreadstat (Stata .dta), PyTorch .pt |
| Environment | Google Colab (CUDA GPU) |

## Requirements

```bash
