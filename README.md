# gemsearch

Fast, offline semantic photo & screenshot search powered by Google's multimodal **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)** (`google/embeddinggemma-2`).

Maps text, images, and interleaved `image + text` queries into a shared 768-dimensional vector space. Tuned for consumer Linux + NVIDIA GPUs (~1–2 GB VRAM in `bf16`), with zero database dependencies and a built-in local Web UI.

### Highlights
- **Multimodal Search:** Search by natural language (`"dog in snow"`), example photo, **composed image + text** (`photo + "at night"`), or visual similarity.
- **OCR-Aware:** Built on Gemma 4—finds both visual scenes and text inside screenshots, receipts, and documents.
- **Zero-DB Atomic Storage:** Stores vectors in a single `.npz` file under `~/.cache/gemsearch/` (never touches your photo folder). Incremental, resumable, and crash-safe.
- **Tunable Speed vs. Detail:** Configurable vision token budgets (`--budget 70..1120`) and search-time Matryoshka dimension truncation (`--dim 768|512|256|128`).
- **Built-in Web UI:** Instant thumbnail grid, drag/drop/paste image search, statistical cliff/MAD score cutoff, and keyboard lightbox navigation.
- **Formats:** `jpg`, `jpeg`, `png`, `webp`, `avif`, `heic`, `heif`, `bmp`, `gif`, `tif`, `tiff`.

---

## Setup

Check your NVIDIA driver version with `nvidia-smi`:
- **Driver 580+**: default PyPI `torch` (CUDA 13)
- **Driver 570–579**: add `--index-url https://download.pytorch.org/whl/cu128` when installing `torch torchvision`

### Option A: `uv`
```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch torchvision
uv pip install --python .venv -r requirements.txt
.venv/bin/python gemsearch.py info
```

### Option B: `conda`
```bash
conda create -n gemsearch python=3.12 -y
conda activate gemsearch
pip install torch torchvision
pip install -r requirements.txt
python gemsearch.py info
```

---

## Usage

```bash
python gemsearch.py index  ~/Pictures                              # build/update index (resumable)
python gemsearch.py search ~/Pictures "sunset over water" -k 10    # text search
python gemsearch.py search ~/Pictures --image ~/query.jpg          # image search
python gemsearch.py search ~/Pictures "in winter" --image ~/q.jpg  # composed image + text search
python gemsearch.py serve  ~/Pictures                              # web UI on http://127.0.0.1:8765
python gemsearch.py bench  ~/Pictures -n 128                       # throughput, VRAM & recall benchmark
```

Pass `--offline` after the initial model download to disable all Hugging Face network calls.

### Key Flags
| Flag | Values | Description |
| :--- | :--- | :--- |
| `--budget` | `70, 140, 280, 560, 1120` | Vision tokens per image (default `280`). Higher = finer detail/OCR, more VRAM. |
| `--dim` | `768, 512, 256, 128` | Matryoshka truncation at search time (one index serves all dims). |
| `--batch` | `int` (default `16`) | GPU batch size; automatically halves on CUDA OOM. |
| `--port` | `int` (default `8765`) | Web UI port (`127.0.0.1` only by default). |

---

## Architecture
- `gemsearch.py` — Model loader (`bfloat16`, audio encoder stripped to save VRAM), threaded CPU image prefetcher, atomic `.npz` indexer, CLI, and benchmark suite.
- `webui.py` — Local `ThreadingHTTPServer` + dependency-free Web UI with Median/MAD cliff filtering, composed image+text search, and localhost security guards (`Host`, `Origin`, `CSP`).