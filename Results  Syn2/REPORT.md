# Controlled Synthetic Benchmark — Text Only vs Text + Metadata

## Purpose
This is a newly generated **synthetic controlled benchmark**, not a real-world dataset. It is designed to test the hypothesis that metadata can add information when text alone is ambiguous. Do not describe these rows or results as collected real customer reviews.

## Dataset
- Rows: 5,000
- Training rows: 4,000
- Test rows: 1,000
- Target: 0 = genuine, 1 = fake
- Features: review text, star rating, review length, sentiment score, rating–sentiment mismatch
- Seed: 42
- Split: stratified 80/20

## Measured results from this generated dataset
          model  accuracy  precision   recall  macro_f1      mcc  roc_auc_auxiliary  average_precision_auxiliary
      Text only     0.618   0.635714 0.538306  0.615285 0.237791           0.680108                     0.693313
Text + metadata     0.730   0.757991 0.669355  0.728819 0.462593           0.773794                     0.790499

## Confusion matrices
          model  TN  FP  FN  TP
      Text only 351 153 229 267
Text + metadata 398 106 164 332

## Interpretation
These values are measured on the generated controlled benchmark. Because the data-generation process intentionally gives metadata noisy class-correlated information, the experiment can demonstrate how metadata may improve classification in a controlled setting. It does not establish performance on real-world reviews.

## Limitations
- Review texts are generated from templates and are not customer-authored.
- The label and metadata relationship is synthetic and should be disclosed.
- Sentiment scores are simulated, not produced by a validated sentiment model or human annotators.
- Results depend on the chosen data-generation mechanism and random seed.
