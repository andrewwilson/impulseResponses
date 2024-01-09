# impulseResponses

Tools to generate and analyses audio signals.

Specifically the generation of impulse responses to equalise one waveform to closely resemble another.

## Approach

Given:
- a set of synchronised samples
- a set of synchronized test/validation samples
- split these into many (optionally overlapping) smaller parts
- consider applying windowing function to each sample to reduce artificial transient noise.
- consider normalising samples to same loudness, or RMS level. 
- compute IR for each sample pair
- score the effectiveness of IRs by applying to source sample and scoring resultant versus the target sample
- can be done on both training data, and on validation samples. 
- Also by cross validation on the sliced samples.
- discard IRs that score very poorly
- blend IRs to form a single resultant IR.
- evaluate on validation samples.
- lots of optional parts involved. compute all and see which approach scores the best.

