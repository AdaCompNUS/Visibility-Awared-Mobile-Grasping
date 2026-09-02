
## Post-hoc amendment (2026-08-30): SG-VLA training details located in paper table captions
The paper does specify (missed in the initial extraction): pick+place models
with segmentation decoders train 5 epochs; global batch 512; Adam CONSTANT
lr 2e-5; 8x A100. Our run (3 epochs, batch 64, AdamW cosine+warmup 2e-5,
4x A5000) delivers ~12.6K gradient updates vs their ~11.7K (5 x 1.2M / 512) —
the same optimization budget at our batch size. Deviations now known rather
than assumed: cosine-with-warmup vs constant LR; single-stage vs their
2+4-epoch progressive stages (their stages serve the multi-subtask setting).
User decision: finish at 3 epochs.
