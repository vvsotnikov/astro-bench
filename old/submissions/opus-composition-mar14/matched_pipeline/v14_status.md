# v14 Pre-training Status

## Current state
- Pass 1: COMPLETE (73.3M events, 4596s)
- Pass 2: IN PROGRESS (run 500+/1308, 100M+ cumulative events)
- Fine-tuning: NOT STARTED
- DE optimization: NOT STARTED

## Timeline
- Started: ~Mar 16 20:00 UTC
- Pass 1 duration: 77 min
- Pass 2 ETA: ~80 min from pass 1 end
- Fine-tuning ETA: ~40 min after pass 2
- DE ETA: ~15 min after fine-tuning
- Total expected: ~3.5 hours from start

## Notes
- Network download speed varies, causing stalls between runs
- Loss plateaued at ~0.286 (down from initial 0.293)
- Using streaming approach (constant 2.5GB memory)
- polish=False in eval_utils to prevent DE hanging
