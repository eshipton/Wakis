Round-2 run data copied from the invrs-wakis results (`round2-full/c55-rerun/shard_0/ceviche_waveguide_bend_seed1`).
design.npy: final design; summary.json: official gym scores, length scales, candidates; log.jsonl: optimization log;
fields_compact.npz: Wakis DFT fields of the final design as float16 normalized by <key>_scale (complex fields split into _re/_im);
snapshots_thinned.npz: every 3rd iteration plus stage boundaries and the last iteration (kept_index = original index).
