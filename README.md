# gemsearch

Local photo search with Google's EmbeddingGemma 2 (text and image in one vector space).
Built for Linux + NVIDIA CUDA, tuned for an 8 GB laptop GPU (RTX 3070 Mobile).

Search by text ("dog playing in snow"), by example image, or "find similar" from any
result. Everything runs locally; after the first model download it can run fully offline.

## Setup

1. Check the driver: `nvidia-smi`. Note the driver version (top line).
2. Create an environment (the commands below work in bash and nushell):

```
uv venv .venv --python 3.12
```

3. Install PyTorch with a CUDA build your driver supports:
   - Driver **580 or newer**: plain PyPI torch works (it ships CUDA 13 on Linux).
     `uv pip install --python .venv torch torchvision`
   - Driver **570-579**: use the CUDA 12.8 wheels.
     `uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   - If unsure, use the selector at https://pytorch.org/get-started/locally/
4. Install the rest:

```
uv pip install --python .venv -r requirements.txt
```

5. Verify:

```
.venv/bin/python gemsearch.py info
```

You want a line like `NVIDIA GeForce RTX 3070 Laptop GPU, 8.0 GB, compute 8.6, ... bf16 ok`.

## Use

```
.venv/bin/python gemsearch.py index  ~/Pictures            # first run downloads the model
.venv/bin/python gemsearch.py search ~/Pictures "sunset over water" -k 10
.venv/bin/python gemsearch.py search ~/Pictures --image ~/query.jpg
.venv/bin/python gemsearch.py serve  ~/Pictures            # web UI at http://127.0.0.1:8765
.venv/bin/python gemsearch.py bench  ~/Pictures -n 128     # speed / VRAM / quality tests
```

Add `--offline` to any command after the first download to stop all Hugging Face Hub
contact. Hub telemetry is disabled by default.

## Testing the model

Two settings change what the model sees and stores:

- `--budget` (70, 140, 280, 560, 1120): vision tokens per image. More tokens capture finer
  detail (small text, distant objects) but index slower and use more VRAM. Default 280.
  Each budget gets its own index file, so you can index the same folder at 140 and 560
  and compare the results directly with `search --budget 140` vs `--budget 560`.
- `--dim` (768, 512, 256, 128): Matryoshka truncation, applied at search time. One index
  serves every dim. Smaller dims mean less storage and faster ranking at large scale, at
  some cost in accuracy. The web UI has a dim dropdown that re-runs the current search.

`bench` runs on a fixed random sample and reports:
1. images/sec and peak VRAM per budget
2. a batch-size sweep (finds the fastest batch that fits in 8 GB)
3. text query latency
4. how closely each budget and dim agree with the most detailed setting (mean cosine and
   top-10 overlap)
5. top 3 hits per test query, so you can eyeball accuracy

Use `--queries "red car;receipt;whiteboard with writing"` to test with your own queries.
Query prompts can be changed with `search --prompt` (default `SearchQuery`; the model also
defines `QuestionAnswering`, `FactChecking`, `Classification`, `Clustering`,
`SentenceSimilarity`).

## How it behaves

- **Index location:** `~/.cache/gemsearch/<folder>-<hash>/`, never inside your photo
  folder. Override with `--index-dir`. `info PHOTOS` lists the indexes and their sizes.
- **Incremental:** re-running `index` embeds only new or changed files (by mtime and size)
  and drops deleted ones. Unreadable files are remembered and not retried until they change.
- **Resumable:** progress is saved every 60 s (`--checkpoint`) and on Ctrl+C. Run the same
  command to continue. Index writes are atomic (single file, temp + rename).
- **GPU memory:** default batch 16. On CUDA out-of-memory the batch halves automatically
  and continues. Images are decoded on CPU threads (`--workers`) while the GPU works.
- **Precision:** bf16 on Ampere and newer, fp32 otherwise. Never fp16; the model returns
  NaN or degraded vectors in fp16. Non-finite outputs are rejected.
- **Skips:** hidden folders and files, symlinks, and non-image files.
- **Formats:** jpg, jpeg, png, webp, avif, heic, heif, bmp, gif (first frame), tif, tiff.

## Web UI security

- Binds `127.0.0.1` only by default and has **no authentication**.
- Checks the `Host` header (blocks DNS-rebinding attacks from web pages you visit).
- Image uploads require an `image/*` content type and a same-origin `Origin`, which blocks
  cross-site form posts.
- Serves only files that are in the index, resolved inside the photo folder.
- Errors are logged on the server; clients get a generic message.

To use it from another machine, prefer an SSH tunnel:
`ssh -L 8765:127.0.0.1:8765 laptop`. If you put it behind a reverse proxy, pass
`--allow-host your.hostname` and add authentication at the proxy.

## Laptop notes (RTX 3070 Mobile)

- Plug into AC power. On battery the GPU is power-limited and `bench` numbers drop
  sharply.
- Hybrid graphics (Optimus) does not need `prime-run` for CUDA compute. Confirm the
  process shows up in `nvidia-smi` while indexing.
- Expect roughly 1 GB of VRAM for the model weights in bf16. The rest goes to activations,
  which grow with `--budget` x `--batch`. If you see repeated out-of-memory messages at
  budget 1120, start with `--batch 4`.

## If CUDA is not available

`info` prints `CUDA NOT available` when the torch build does not match the driver (for
example CUDA 13 torch with a 570-series driver). Reinstall torch from the cu128 index (step
3), or update the driver. `--device cpu` works but is slow.

## Files

- `gemsearch.py`: model loading, indexing, search, bench, CLI
- `webui.py`: the local web server and page
