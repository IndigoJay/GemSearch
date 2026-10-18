#!/usr/bin/env python3
"""gemsearch - local image search with Google EmbeddingGemma 2.

Built for Linux + NVIDIA CUDA, tuned for 8 GB laptop GPUs (RTX 3070 Mobile).

    gemsearch.py info   [PHOTOS]                 GPU / library check, index stats
    gemsearch.py index  PHOTOS                   build or update the index (resumable)
    gemsearch.py search PHOTOS "dog in snow"     text search
    gemsearch.py search PHOTOS --image q.jpg     search by example image
    gemsearch.py serve  PHOTOS                   web UI on http://127.0.0.1:8765
    gemsearch.py bench  PHOTOS                   speed / VRAM / quality tests

The index lives in ~/.cache/gemsearch/ (never inside the photo folder), one file per
model + vision budget, so different budgets can be compared side by side. Full
768-dim vectors are stored; smaller Matryoshka dims (512/256/128) are applied at
search time, so one index serves every --dim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

DEFAULT_MODEL = "google/embeddinggemma-2"
FULL_DIM = 768
DIMS = (768, 512, 256, 128)               # Matryoshka sizes
BUDGETS = (70, 140, 280, 560, 1120)        # vision soft tokens per image
DEFAULT_BUDGET = 280                       # model default
DEFAULT_BATCH = 16                         # safe on 8 GB at budget 280; halves itself on OOM
MAX_SIDE = 2048                            # pre-shrink huge photos (processor downsizes anyway)
CHECKPOINT_SECS = 60
INDEX_VERSION = 1
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif", ".bmp", ".gif", ".tif", ".tiff"}
BENCH_QUERIES = (
    "a person smiling", "a dog", "a cat", "a beach or the ocean", "food on a plate",
    "a car on a road", "a screenshot of text", "mountains and trees", "a group of people",
    "a city at night",
)


def log(*args, end="\n"):
    print(*args, file=sys.stderr, end=end, flush=True)


class UserError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.msg, self.code = msg, code


# ----------------------------------------------------------------- torch helpers
# torch is imported lazily so `--help`, `info` without CUDA, and tests stay fast.

def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda():
    t = _torch()
    return t if t is not None and t.cuda.is_available() else None


def oom_errors():
    t = _torch()
    if t is None:
        return ()
    return tuple({getattr(t, "OutOfMemoryError", t.cuda.OutOfMemoryError), t.cuda.OutOfMemoryError})


def cuda_sync():
    t = _cuda()
    if t:
        t.cuda.synchronize()


def cuda_free():
    t = _cuda()
    if t:
        t.cuda.empty_cache()


def vram_reset_peak():
    t = _cuda()
    if t:
        t.cuda.reset_peak_memory_stats()


def vram_peak_gb():
    t = _cuda()
    return t.cuda.max_memory_allocated() / 2**30 if t else float("nan")


def vram_now_gb():
    t = _cuda()
    return t.cuda.memory_allocated() / 2**30 if t else float("nan")


def gpu_description():
    t = _torch()
    if t is None:
        return "PyTorch not installed"
    if not t.cuda.is_available():
        return f"torch {t.__version__}: CUDA NOT available (built for CUDA {t.version.cuda})"
    p = t.cuda.get_device_properties(0)
    bf16 = "bf16 ok" if t.cuda.is_bf16_supported() else "no bf16 -> fp32"
    return (f"{p.name}, {p.total_memory / 2**30:.1f} GB, compute {p.major}.{p.minor}, "
            f"torch {t.__version__} / CUDA {t.version.cuda}, {bf16}")


# ----------------------------------------------------------------- model

def load_model(model_id=DEFAULT_MODEL, device="cuda"):
    torch = _torch()
    if torch is None:
        raise SystemExit("PyTorch is not installed - see README.")
    from sentence_transformers import SentenceTransformer

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available to PyTorch.\n"
                "  - check the driver with `nvidia-smi`\n"
                "  - make sure torch's CUDA build matches the driver (see README)\n"
                "  - or run on CPU with --device cpu (slow)")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = torch.float32
    # Never float16: EmbeddingGemma 2 silently returns NaN / degraded vectors in fp16.
    t0 = time.time()
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"dtype": dtype},
        config_kwargs={"audio_config": None},   # text + image only (~440M params)
    )
    log(f"Loaded {model_id} ({str(dtype).replace('torch.', '')}) on "
        f"{gpu_description() if device.startswith('cuda') else device} in {time.time() - t0:.1f}s")
    return model


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def truncate(v, dim):
    """Matryoshka truncation: keep the first `dim` values and re-normalize."""
    return normalize(v[..., :dim]) if dim < v.shape[-1] else v


def _check(out):
    out = np.asarray(out, dtype=np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("model returned non-finite values")
    return normalize(out)


def encode_query(model, text, prompt="SearchQuery"):
    out = model.encode([text], prompt_name=prompt, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    return _check(out)[0]


class Encoder:
    """Image encoder that halves its batch size on CUDA out-of-memory."""

    def __init__(self, model, budget=DEFAULT_BUDGET, batch=DEFAULT_BATCH):
        self.model, self.budget, self.batch = model, budget, max(1, batch)

    def images(self, images):
        while True:
            try:
                out = self.model.encode(
                    [{"image": im} for im in images],   # images take no task prefix
                    batch_size=self.batch,
                    processing_kwargs={"image": {"max_soft_tokens": self.budget}},
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return _check(out)
            except oom_errors():
                cuda_free()
                if self.batch == 1:
                    raise
                self.batch = max(1, self.batch // 2)
                log(f"\nCUDA out of memory - batch size now {self.batch}")

    def compose(self, image, text):
        """Interleaved image + text query (EmbeddingGemma 2 multimodal composition)."""
        txt = text.strip()
        if "<|image|>" not in txt:
            txt = f"<|image|> {txt}"
        out = self.model.encode(
            [{"text": txt, "image": image}],
            batch_size=1,
            processing_kwargs={"image": {"max_soft_tokens": self.budget}},
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _check(out)[0]

    def images_robust(self, items):
        """items: [(key, PIL image)] -> (ok [(key, vec)], failed [(key, error)]).
        If a batch fails for a non-memory reason, retry one by one to isolate bad files.
        Unrecoverable OOM (at batch=1) is re-raised so valid files are never poisoned as unreadable."""
        if not items:
            return [], []
        try:
            vecs = self.images([im for _, im in items])
            return [(k, v) for (k, _), v in zip(items, vecs)], []
        except oom_errors():
            cuda_free()
            raise
        except Exception as e:  # noqa: BLE001
            if len(items) == 1:
                cuda_free()
                return [], [(items[0][0], e)]
        ok, bad = [], []
        for it in items:
            o, b = self.images_robust([it])
            ok += o
            bad += b
        return ok, bad


# ----------------------------------------------------------------- files

def need_folder(p):
    folder = Path(p).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    return folder


def scan(folder):
    """[(rel, path, (mtime_ns, size))] for image files; skips hidden dirs/files and symlinks."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith(".") or Path(name).suffix.lower() not in EXTS:
                continue
            p = Path(root, name)
            try:
                if p.is_symlink():
                    continue
                st = p.stat()
            except OSError:
                continue
            out.append((p.relative_to(folder).as_posix(), p, (st.st_mtime_ns, st.st_size)))
    return out


def load_image(src, max_side=MAX_SIDE):
    """Open a path or file object -> upright RGB PIL image no larger than max_side."""
    with Image.open(src) as im:
        try:
            im.draft("RGB", (max_side, max_side))   # fast JPEG downscale on decode
        except Exception:  # noqa: BLE001
            pass
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


# ----------------------------------------------------------------- index

def cache_root():
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gemsearch"


def index_base(folder, index_dir=None):
    if index_dir:
        return Path(index_dir).expanduser().resolve()
    h = hashlib.sha1(str(folder).encode()).hexdigest()[:10]
    return cache_root() / f"{folder.name or 'root'}-{h}"


def index_path(folder, model_id, budget, index_dir=None):
    return index_base(folder, index_dir) / f"{model_id.replace('/', '--')}__b{budget}.npz"


@dataclass
class Index:
    path: Path
    model: str
    budget: int
    folder: str
    rels: list = field(default_factory=list)
    stamps: list = field(default_factory=list)     # [mtime_ns, size] per row
    failed: dict = field(default_factory=dict)     # rel -> [mtime_ns, size] of unreadable files
    vecs: np.ndarray = field(default_factory=lambda: np.zeros((0, FULL_DIM), np.float32))
    saved: float = 0.0


def save_index(idx):
    """Single-file, atomic write (temp file + rename), so a crash never leaves a torn index."""
    idx.path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"version": INDEX_VERSION, "model": idx.model, "budget": idx.budget,
            "folder": idx.folder, "rels": idx.rels, "stamps": [list(s) for s in idx.stamps],
            "failed": idx.failed, "saved": time.time()}
    tmp = idx.path.with_name(idx.path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, vectors=np.asarray(idx.vecs, np.float32), meta=np.array(json.dumps(meta)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, idx.path)


def load_index(path):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        vecs = np.asarray(z["vectors"], np.float32)
    if meta.get("version") != INDEX_VERSION or len(meta["rels"]) != len(vecs):
        log(f"Ignoring incompatible index {path}")
        return None
    return Index(path, meta["model"], meta["budget"], meta["folder"], meta["rels"],
                 [tuple(s) for s in meta["stamps"]], meta.get("failed", {}), vecs,
                 meta.get("saved", 0.0))


def require_index(folder, a):
    path = index_path(folder, a.model, a.budget, a.index_dir)
    idx = load_index(path)
    if idx is None or not idx.rels:
        raise SystemExit(f"No index at {path}\nRun: gemsearch.py index {folder} --budget {a.budget}")
    return idx


def rank(vecs, q, k, exclude=None):
    """vecs and q already truncated/normalized to the same dim -> [(row, score)]."""
    s = vecs @ q
    if exclude is not None:
        s[exclude] = -np.inf
    k = max(0, min(k, len(s) - (exclude is not None)))
    if k == 0:
        return []
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top])]
    return [(int(i), float(s[i])) for i in top]


# ----------------------------------------------------------------- commands

def cmd_info(a):
    t = _torch()
    log(f"GPU:       {gpu_description()}")
    for mod in ("transformers", "sentence_transformers"):
        try:
            m = __import__(mod)
            log(f"{mod + ':':<22}{m.__version__}")
        except ImportError:
            log(f"{mod + ':':<22}NOT INSTALLED")
    if t is not None and not t.cuda.is_available():
        log("\nCUDA is not usable. See README > 'If CUDA is not available'.")
    if not a.folder:
        return
    folder = need_folder(a.folder)
    base = index_base(folder, a.index_dir)
    log(f"\nIndex dir: {base}")
    for p in sorted(base.glob("*.npz")) if base.exists() else []:
        idx = load_index(p)
        if idx:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(idx.saved))
            log(f"  {p.name}: {len(idx.rels)} images, {len(idx.failed)} unreadable, "
                f"budget {idx.budget}, saved {when}, {p.stat().st_size / 2**20:.1f} MB")


def cmd_index(a):
    folder = need_folder(a.folder)
    path = index_path(folder, a.model, a.budget, a.index_dir)
    old = load_index(path)
    old_rows = {r: (tuple(s), i) for i, (r, s) in enumerate(zip(old.rels, old.stamps))} if old else {}
    old_failed = old.failed if old else {}

    files = scan(folder)
    present = {rel for rel, _, _ in files}
    rels, stamps, kept, todo, failed = [], [], [], [], {}
    for rel, p, st in files:
        prev = old_rows.get(rel)
        if prev and prev[0] == st:
            rels.append(rel)
            stamps.append(st)
            kept.append(prev[1])
        elif tuple(old_failed.get(rel, ())) == st:
            failed[rel] = list(st)            # unchanged unreadable file: don't retry
        else:
            todo.append((rel, p, st))
    removed = len(set(old_rows) - present)
    log(f"{len(files)} images in {folder}: {len(rels)} up to date, {len(todo)} to embed, "
        f"{removed} removed, {len(failed)} known unreadable")
    log(f"Index: {path}")

    base_vecs = old.vecs[kept] if kept else np.zeros((0, FULL_DIM), np.float32)
    new_vecs = []

    def flush():
        vecs = np.vstack([base_vecs] + ([np.stack(new_vecs)] if new_vecs else []))
        save_index(Index(path, a.model, a.budget, str(folder), list(rels), list(stamps), failed, vecs))

    if not todo:
        if removed or old is None or failed != old_failed:
            flush()
        log("Nothing to embed.")
        return

    model = load_model(a.model, a.device)
    enc = Encoder(model, a.budget, a.batch)
    group = max(a.batch, 1)
    groups = [todo[i:i + group] for i in range(0, len(todo), group)]
    tty = sys.stderr.isatty()

    def decode(item):
        rel, p, st = item
        try:
            return rel, st, load_image(p), None
        except Exception as e:  # noqa: BLE001
            return rel, st, None, e

    done = 0
    t_start = last_save = time.time()
    vram_reset_peak()
    pool = ThreadPoolExecutor(max_workers=a.workers)
    try:
        nxt = [pool.submit(decode, it) for it in groups[0]]
        for gi in range(len(groups)):
            cur = [f.result() for f in nxt]                     # decoded on CPU threads...
            nxt = [pool.submit(decode, it) for it in groups[gi + 1]] if gi + 1 < len(groups) else []
            good = []                                           # ...while the GPU works on this group
            for rel, st, im, err in cur:
                if err is None:
                    good.append(((rel, st), im))
                else:
                    failed[rel] = list(st)
                    log(f"\nskip {rel}: {err}")
            ok, bad = enc.images_robust(good)
            for (rel, st), v in ok:
                rels.append(rel)
                stamps.append(st)
                new_vecs.append(v)
            for (rel, st), e in bad:
                failed[rel] = list(st)
                log(f"\nskip {rel}: {type(e).__name__}: {e}")

            done += len(cur)
            el = time.time() - t_start
            rate = done / el if el else 0.0
            eta = (len(todo) - done) / rate if rate else 0
            line = (f"[{done}/{len(todo)}] {rate:.1f} img/s  batch {enc.batch}  "
                    f"VRAM peak {vram_peak_gb():.2f} GB  ETA {eta / 60:.1f} min")
            if tty:
                log("\r" + line + "   ", end="")
            elif gi % 20 == 0 or gi == len(groups) - 1:
                log(line)
            if time.time() - last_save >= a.checkpoint:
                flush()
                last_save = time.time()
        flush()
    except oom_errors():
        pool.shutdown(wait=False, cancel_futures=True)
        flush()
        log(f"\nCUDA out of memory even at batch=1 (budget {a.budget}). "
            f"Saved {len(rels)} images; retry with a smaller --budget.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        flush()
        log(f"\nInterrupted - saved {len(rels)} images. Run the same command to resume.")
        raise SystemExit(130)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    el = time.time() - t_start
    log(f"\nDone: {len(new_vecs)} embedded in {el:.0f}s ({len(new_vecs) / max(el, 1e-9):.1f} img/s), "
        f"{len(rels)} total, {len(failed)} unreadable.")


def cmd_search(a):
    folder = need_folder(a.folder)
    if not (a.query or a.image):
        raise SystemExit("Give a text query, --image FILE, or both (multimodal query)")
    idx = require_index(folder, a)
    model = load_model(a.model, a.device)
    t0 = time.time()
    if a.image and a.query:
        q = Encoder(model, idx.budget, 1).compose(load_image(a.image), a.query)
    elif a.image:
        q = Encoder(model, idx.budget, 1).images([load_image(a.image)])[0]
    else:
        q = encode_query(model, a.query, a.prompt)
    t1 = time.time()
    hits = rank(truncate(idx.vecs, a.dim), truncate(q, a.dim), a.k)
    t2 = time.time()
    for row, score in hits:
        print(f"{score:.3f}  {folder / idx.rels[row]}")
    log(f"encode {1000 * (t1 - t0):.0f} ms, rank {1000 * (t2 - t1):.1f} ms over "
        f"{len(idx.rels)} images (dim {a.dim}, budget {idx.budget})")


def _overlap(a_hits, b_hits):
    a_set, b_set = {r for r, _ in a_hits}, {r for r, _ in b_hits}
    return len(a_set & b_set) / max(1, len(b_set))


def cmd_bench(a):
    folder = need_folder(a.folder)
    files = scan(folder)
    if not files:
        raise SystemExit("No images found.")
    sample = random.Random(0).sample(files, min(a.n, len(files)))
    budgets = sorted({int(x) for x in a.budgets.split(",")})
    batches = sorted({int(x) for x in a.batches.split(",")}) if a.batches else []
    queries = [q.strip() for q in a.queries.split(";")] if a.queries else list(BENCH_QUERIES)
    bad = [b for b in budgets if b not in BUDGETS]
    if bad:
        raise SystemExit(f"Budgets must be from {BUDGETS}")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        decoded = list(pool.map(lambda f: (f[0], _try_load(f[1])), sample))
    imgs = [(rel, im) for rel, im in decoded if im is not None]
    names = [rel for rel, _ in imgs]
    images = [im for _, im in imgs]
    log(f"Decoded {len(images)} sample images in {time.time() - t0:.1f}s (decode time excluded below)")

    vram_reset_peak()
    model = load_model(a.model, a.device)
    print(f"\nGPU: {gpu_description()}")
    print(f"Model weights in VRAM: {vram_now_gb():.2f} GB")

    # warm-up (CUDA kernels, allocator)
    Encoder(model, DEFAULT_BUDGET, 4).images(images[:4])
    encode_query(model, "warm up")

    def timed(budget, batch):
        enc = Encoder(model, budget, batch)
        cuda_free()
        vram_reset_peak()
        cuda_sync()
        t = time.time()
        e = enc.images(images)
        cuda_sync()
        return e, len(images) / (time.time() - t), vram_peak_gb(), enc.batch

    print(f"\n1) Image embedding throughput, {len(images)} images, batch {a.batch}")
    print(f"   {'budget':>6} {'img/s':>8} {'peak VRAM':>10} {'batch used':>11}")
    embs = {}
    for b in budgets:
        e, rate, peak, used = timed(b, a.batch)
        embs[b] = e
        print(f"   {b:>6} {rate:>8.1f} {peak:>8.2f} GB {used:>11}")

    if batches:
        print(f"\n2) Batch size sweep at budget {DEFAULT_BUDGET}")
        print(f"   {'batch':>6} {'img/s':>8} {'peak VRAM':>10}")
        for bs in batches:
            _, rate, peak, used = timed(DEFAULT_BUDGET, bs)
            note = "" if used == bs else f"  (OOM -> {used})"
            print(f"   {bs:>6} {rate:>8.1f} {peak:>8.2f} GB{note}")

    lat, qvecs = [], []
    for q in queries:
        cuda_sync()
        t = time.time()
        qvecs.append(encode_query(model, q))
        cuda_sync()
        lat.append(1000 * (time.time() - t))
    print(f"\n3) Text query latency: median {statistics.median(lat):.1f} ms, max {max(lat):.1f} ms "
          f"({len(queries)} queries)")

    ref_b = max(budgets)
    ref = embs[ref_b]
    k = min(10, len(images))
    print(f"\n4) Agreement with budget {ref_b} @ 768 dims (higher = closer to the most detailed setting)")
    print(f"   {'budget':>6} {'mean cos':>9} {'top-' + str(k) + ' overlap':>14}")
    ref_hits = [rank(ref, q, k) for q in qvecs]
    for b in budgets:
        cos = float(np.mean(np.sum(embs[b] * ref, axis=1)))
        ov = statistics.mean(_overlap(rank(embs[b], q, k), h) for q, h in zip(qvecs, ref_hits))
        print(f"   {b:>6} {cos:>9.3f} {ov:>14.0%}")
    print(f"\n   {'dim':>6} {'top-' + str(k) + ' overlap':>24}   (Matryoshka truncation, budget {ref_b})")
    for d in DIMS:
        ov = statistics.mean(_overlap(rank(truncate(ref, d), truncate(q, d), k), h)
                             for q, h in zip(qvecs, ref_hits))
        print(f"   {d:>6} {ov:>24.0%}")

    print(f"\n5) Eyeball check - top 3 per query (budget {ref_b}, dim 768):")
    for q, hits in zip(queries, ref_hits):
        print(f"   {q!r}")
        for row, score in hits[:3]:
            print(f"      {score:.3f}  {names[row]}")


def _try_load(p):
    try:
        return load_image(p)
    except Exception:  # noqa: BLE001
        return None


def cmd_serve(a):
    from webui import serve
    serve(a)


# ----------------------------------------------------------------- CLI

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path (default %(default)s)")
    common.add_argument("--budget", type=int, default=DEFAULT_BUDGET, choices=BUDGETS,
                        help="vision tokens per image: more = finer detail, slower (default %(default)s)")
    common.add_argument("--device", default="cuda", help="cuda, cuda:1, or cpu (default %(default)s)")
    common.add_argument("--index-dir", help="where to keep the index (default ~/.cache/gemsearch/<folder>-<hash>)")
    common.add_argument("--offline", action="store_true", help="never contact the Hugging Face Hub (after first download)")
    common.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="image decode threads")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", parents=[common], help="check GPU/libraries and list indexes")
    p.add_argument("folder", nargs="?")

    p = sub.add_parser("index", parents=[common], help="build or update an index (resumable)")
    p.add_argument("folder")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="images per GPU batch (default %(default)s)")
    p.add_argument("--checkpoint", type=int, default=CHECKPOINT_SECS, help="save progress every N seconds")

    p = sub.add_parser("search", parents=[common], help="search by text or example image")
    p.add_argument("folder")
    p.add_argument("query", nargs="?")
    p.add_argument("--image", help="search by example image instead of text")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--dim", type=int, default=FULL_DIM, choices=DIMS)
    p.add_argument("--prompt", default="SearchQuery",
                   help="query prompt name (SearchQuery, QuestionAnswering, Classification, ...)")

    p = sub.add_parser("serve", parents=[common], help="web UI")
    p.add_argument("folder")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default %(default)s; no auth - keep it local)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--allow-host", action="append", default=[],
                   help="extra Host header name to accept, e.g. a reverse-proxy hostname (repeatable)")
    p.add_argument("--dim", type=int, default=FULL_DIM, choices=DIMS, help="default dim for API calls")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--verbose", action="store_true", help="log each request")

    p = sub.add_parser("bench", parents=[common], help="speed, VRAM and quality tests on a sample")
    p.add_argument("folder")
    p.add_argument("-n", type=int, default=128, help="sample size (default %(default)s)")
    p.add_argument("--budgets", default="70,140,280,560,1120")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--batches", default="4,8,16,32", help="batch sizes to sweep ('' to skip)")
    p.add_argument("--queries", help="semicolon-separated text queries for the quality checks")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    # Must be set before torch / huggingface libraries are imported.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 8 GB
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    {"info": cmd_info, "index": cmd_index, "search": cmd_search,
     "serve": cmd_serve, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
