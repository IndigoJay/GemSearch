"""Web UI for gemsearch - start it with `gemsearch.py serve PHOTOS`.

Security model: binds 127.0.0.1 by default, has no authentication, checks the Host
header (blocks DNS rebinding), checks Origin and Content-Type on uploads (blocks
cross-site form posts), and only serves files that are in the index.
"""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import gemsearch as gs

MAX_UPLOAD = 30 * 1024 * 1024
THUMB_SIZE = 320
RELOAD_CHECK_SECS = 2.0


class Snapshot:
    """Immutable view of one index version; swapped atomically when the file changes."""

    def __init__(self, idx):
        self.idx = idx
        self.row_by_rel = {r: i for i, r in enumerate(idx.rels)}
        self._by_dim = {gs.FULL_DIM: idx.vecs}
        self._lock = threading.Lock()

    def vecs(self, dim):
        with self._lock:
            if dim not in self._by_dim:
                self._by_dim[dim] = gs.truncate(self.idx.vecs, dim)
            return self._by_dim[dim]


class App:
    def __init__(self, a, folder):
        self.a, self.folder = a, folder
        self.path = gs.index_path(folder, a.model, a.budget, a.index_dir)
        self.thumb_dir = self.path.parent / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.reload_lock = threading.Lock()
        self.model_lock = threading.Lock()       # one GPU call at a time
        self.snap, self._stamp, self._checked = None, None, 0.0
        self.snapshot()                          # fail fast if there is no index
        self.model = gs.load_model(a.model, a.device)
        self.encoder = gs.Encoder(self.model, self.snap.idx.budget, 1)

    def snapshot(self):
        if self.snap is not None and time.monotonic() - self._checked < RELOAD_CHECK_SECS:
            return self.snap
        with self.reload_lock:
            self._checked = time.monotonic()
            try:
                st = self.path.stat()
            except FileNotFoundError:
                if self.snap is not None:
                    return self.snap
                raise SystemExit(f"No index at {self.path}\nRun: gemsearch.py index {self.folder} "
                                 f"--budget {self.a.budget}")
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp != self._stamp:
                idx = gs.load_index(self.path)
                if idx is None or not idx.rels:
                    if self.snap is not None:
                        return self.snap
                    raise SystemExit(f"Index {self.path} is empty or unreadable.")
                self.snap, self._stamp = Snapshot(idx), stamp
                gs.log(f"Index loaded: {len(idx.rels)} images (budget {idx.budget})")
        return self.snap

    def file_for(self, snap, rel):
        """Resolve a relative path only if it is an indexed file inside the folder."""
        if rel not in snap.row_by_rel:
            return None
        p = (self.folder / rel).resolve()
        if not p.is_relative_to(self.folder) or not p.is_file():
            return None
        return p

    def thumb(self, p):
        st = p.stat()
        key = hashlib.sha1(f"{p}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
        cache = self.thumb_dir / f"{key}.jpg"
        if cache.exists():
            return cache.read_bytes()
        im = gs.load_image(p, max_side=THUMB_SIZE * 2)
        im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        data = buf.getvalue()
        tmp = cache.with_name(f"{cache.name}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
        return data

    def info(self):
        snap = self.snapshot()
        return {"folder": str(self.folder), "model": snap.idx.model, "budget": snap.idx.budget,
                "count": len(snap.idx.rels), "dims": list(gs.DIMS), "gpu": gs.gpu_description()}

    def search(self, qvec_fn, k, dim, exclude_rel=None):
        snap = self.snapshot()
        t0 = time.perf_counter()
        with self.model_lock:
            q = qvec_fn(snap)
        t1 = time.perf_counter()
        exclude = snap.row_by_rel.get(exclude_rel) if exclude_rel else None
        vecs = snap.vecs(dim)
        q_trunc = gs.truncate(q, dim)
        s = vecs @ q_trunc
        if exclude is not None:
            valid = np.delete(s, exclude)
        else:
            valid = s
        if len(valid):
            med = float(np.median(valid))
            mad = max(float(np.median(np.abs(valid - med))) * 1.4826, 1e-6)
        else:
            med, mad = 0.0, 1.0
        hits = gs.rank(vecs, q_trunc, k, exclude)
        t2 = time.perf_counter()
        return {
            "results": [
                {"rel": snap.idx.rels[r], "score": sc, "z": round((sc - med) / mad, 2)}
                for r, sc in hits
            ],
            "baseline": {"med": round(med, 3), "mad": round(mad, 3)},
            "count": len(snap.idx.rels), "dim": dim,
            "encode_ms": round(1000 * (t1 - t0), 1), "rank_ms": round(1000 * (t2 - t1), 2),
        }

def make_handler(app, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gemsearch"
        sys_version = ""

        def log_message(self, fmt, *args):
            if app.a.verbose:
                gs.log("%s %s" % (self.address_string(), fmt % args))

        # ---- responses
        def _send(self, code, body, ctype, cache="no-store", extra=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj), "application/json")

        # ---- request helpers
        def _int(self, qs, name, default, lo, hi):
            raw = qs.get(name, [str(default)])[0]
            try:
                return max(lo, min(int(raw), hi))
            except ValueError:
                raise gs.UserError(f"'{name}' must be an integer") from None

        def _dim(self, qs):
            d = self._int(qs, "dim", app.a.dim, 1, gs.FULL_DIM)
            if d not in gs.DIMS:
                raise gs.UserError(f"dim must be one of {list(gs.DIMS)}")
            return d

        def _guard(self, fn):
            host = (self.headers.get("Host") or "").lower()
            if host not in allowed_hosts:
                return self._json(403, {"error": "Host not allowed"})
            try:
                fn()
            except gs.UserError as e:
                self._json(e.code, {"error": e.msg})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                try:
                    self._json(500, {"error": "internal error - see server log"})
                except OSError:
                    pass

        def do_GET(self):
            self._guard(self._get)

        def do_POST(self):
            self._guard(self._post)

        # ---- routes
        def _get(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8", extra={"Content-Security-Policy": CSP})
            if u.path == "/api/info":
                return self._json(200, app.info())
            if u.path in ("/thumb", "/img"):
                p = app.file_for(app.snapshot(), qs.get("p", [""])[0])
                if p is None:
                    raise gs.UserError("not found", 404)
                if u.path == "/thumb":
                    return self._send(200, app.thumb(p), "image/jpeg", cache="private, max-age=86400")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                return self._send(200, p.read_bytes(), ctype, cache="private, max-age=3600")
            if u.path == "/api/browse":
                k, dim = self._int(qs, "k", 24, 1, 500), self._dim(qs)
                snap = app.snapshot()
                rels = snap.idx.rels[:k]
                return self._json(200, {
                    "results": [{"rel": r, "score": None} for r in rels],
                    "count": len(snap.idx.rels), "dim": dim,
                    "encode_ms": 0.0, "rank_ms": 0.0,
                })
            if u.path == "/api/search":
                q = qs.get("q", [""])[0].strip()
                if not q:
                    raise gs.UserError("empty query")
                if len(q) > 2000:
                    raise gs.UserError("query too long")
                k, dim = self._int(qs, "k", 24, 1, 500), self._dim(qs)
                return self._json(200, app.search(lambda s: gs.encode_query(app.model, q), k, dim))
            if u.path == "/api/similar":
                rel = qs.get("p", [""])[0]
                q = qs.get("q", [""])[0].strip()
                if len(q) > 2000:
                    raise gs.UserError("query too long")
                k, dim = self._int(qs, "k", 24, 1, 500), self._dim(qs)
                snap = app.snapshot()
                if rel not in snap.row_by_rel:
                    raise gs.UserError("image not in index", 404)
                if q:
                    p = app.file_for(snap, rel)
                    if p is None:
                        raise gs.UserError("image not found on disk", 404)
                    im = gs.load_image(p)
                    return self._json(200, app.search(lambda s: app.encoder.compose(im, q), k, dim, rel))
                return self._json(200, app.search(lambda s: s.idx.vecs[s.row_by_rel[rel]], k, dim, rel))
            raise gs.UserError("not found", 404)

        def _post(self):
            u = urlparse(self.path)
            if u.path != "/api/search_image":
                raise gs.UserError("not found", 404)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc.lower() not in allowed_hosts:
                raise gs.UserError("cross-origin request refused", 403)
            if not (self.headers.get("Content-Type") or "").startswith("image/"):
                raise gs.UserError("send the image bytes with an image/* Content-Type", 415)
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise gs.UserError("bad Content-Length") from None
            if n <= 0 or n > MAX_UPLOAD:
                raise gs.UserError(f"upload must be 1 byte to {MAX_UPLOAD // 2**20} MB", 413)
            qs = parse_qs(u.query)
            q = qs.get("q", [""])[0].strip()
            if len(q) > 2000:
                raise gs.UserError("query too long")
            k, dim = self._int(qs, "k", 24, 1, 500), self._dim(qs)
            data = self.rfile.read(n)
            try:
                im = gs.load_image(io.BytesIO(data))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
                raise gs.UserError(f"could not read image: {e}") from None
            fn = (lambda s: app.encoder.compose(im, q)) if q else (lambda s: app.encoder.images([im])[0])
            return self._json(200, app.search(fn, k, dim))
    return Handler


def serve(a):
    if not hasattr(a, "dim"):
        a.dim = gs.FULL_DIM
    folder = gs.need_folder(a.folder)
    app = App(a, folder)
    names = {"127.0.0.1", "localhost"} | {h.lower() for h in a.allow_host}
    if a.host not in ("127.0.0.1", "localhost"):
        names.add(a.host.lower())
        gs.log(f"WARNING: listening on {a.host} with NO authentication. Anyone who can reach "
               f"this port can search and view the indexed photos.")
    allowed = {f"{n}:{a.port}" for n in names}
    if a.port == 80:
        allowed |= names
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((a.host, a.port), make_handler(app, allowed))
    url = f"http://127.0.0.1:{a.port}/" if a.host in ("127.0.0.1", "0.0.0.0") else f"http://{a.host}:{a.port}/"
    gs.log(f"Serving {folder}\nOpen {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


CSP = ("default-src 'none'; img-src 'self' blob:; style-src 'unsafe-inline'; "
       "script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gemsearch</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#76766f;--card:#fff;--line:#d9d9d3;--acc:#1f6f5c}
@media (prefers-color-scheme:dark){:root{--bg:#141514;--fg:#e9e9e4;--mut:#9a9a92;--card:#1e201f;--line:#343633;--acc:#4fb79b}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
form{display:flex;gap:8px;flex-wrap:wrap}
input[type=search]{flex:1 1 260px;min-width:0;padding:9px 12px;font:inherit;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
select,button{padding:9px 12px;font:inherit;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.go{background:var(--acc);border-color:var(--acc);color:#fff}
label{display:flex;align-items:center;gap:4px;color:var(--mut);font-size:13px}
#drop.over{outline:2px dashed var(--acc)}
#meta{margin-top:8px;font-size:13px;color:var(--mut);display:flex;gap:12px;flex-wrap:wrap;align-items:center}
#qimg{height:28px;border-radius:4px;display:none}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;padding:14px 16px 32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden;cursor:pointer;padding:0;text-align:left}
.card img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:var(--line)}
.card div{padding:5px 7px;font-size:12px}
.card .n{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .s{color:var(--mut)}
#lb{position:fixed;inset:0;background:#000e;display:none;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:16px;z-index:10}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:82vh;object-fit:contain}
#lb p{margin:0;color:#ddd;font-size:13px;word-break:break-all;text-align:center}
</style></head><body>
<header>
<form id="f">
  <input id="q" type="search" placeholder="Describe a photo (or add text modifier to an image)" autofocus>
  <button class="go">Search</button>
  <button type="button" id="drop" title="Choose, drop or paste an image">By image...</button>
  <button type="button" id="brw" title="Browse indexed photos">Browse</button>
  <input id="file" type="file" accept="image/*" hidden>
  <label>cutoff <select id="cut" title="Filter tail results using largest score drop (cliff) or robust Median/MAD z-score">
    <option value="cliff" selected>Auto (cliff)</option>
    <option value="3.5">Strict (&gt;3.5σ)</option>
    <option value="2.5">Medium (&gt;2.5σ)</option>
    <option value="1.8">Loose (&gt;1.8σ)</option>
    <option value="0">Off (all)</option>
  </select></label>
  <label>dim <select id="dim"><option>768</option><option>512</option><option>256</option><option>128</option></select></label>
  <label>show <select id="k"><option>24</option><option>48</option><option>96</option><option>200</option></select></label>
</form>
<div id="meta"><img id="qimg" title="Click to clear image filter" style="cursor:pointer" alt=""><span id="status">Loading...</span></div>
</header>
<main id="grid"></main>
<div id="lb"><img id="lbimg" alt=""><p id="lbcap"></p>
  <div><button id="prv" title="Previous (Left Arrow)">&#9664; Prev</button> <button id="sim">Find similar</button> <button id="cls">Close</button> <button id="nxt" title="Next (Right Arrow)">Next &#9654;</button></div></div>
<script>
const $ = id => document.getElementById(id);
let info = null, current = null, currentIdx = -1, shownList = [], lastRun = null, lastData = null, lastLabel = '', activeFile = null, activeRel = null;
const status = t => { $('status').textContent = t; };
const params = extra => new URLSearchParams({k: $('k').value, dim: $('dim').value, ...extra});

function clearImage() {
  activeFile = null; activeRel = null;
  $('qimg').style.display = 'none'; $('qimg').removeAttribute('src');
  $('file').value = '';
}

function browse() {
  clearImage();
  run('browse', () => fetch('/api/browse?' + params({})));
}

fetch('/api/info').then(r => r.json()).then(j => {
  info = j;
  browse();
}).catch(() => status('Could not reach the server.'));

async function run(label, req) {
  lastRun = [label, req];
  status('Searching...');
  try {
    const r = await req();
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    lastData = j; lastLabel = label;
    render(j, label);
  } catch (e) { status('Error: ' + e.message); }
}

function applyCutoff(all, base, mode) {
  if (!base || mode === '0' || all.length <= 1 || all[0].z == null) return all;
  if (mode === 'cliff') {
    const above = all.filter(r => r.z >= 2.2);
    const pool = above.length >= 2 ? above : all;
    let bestIdx = -1, maxDrop = 0;
    for (let i = 0; i < pool.length - 1; i++) {
      const drop = pool[i].score - pool[i + 1].score;
      if (drop > maxDrop) { maxDrop = drop; bestIdx = i; }
    }
    if (bestIdx >= 0 && maxDrop >= 0.5 * (base.mad || 0.02)) return pool.slice(0, bestIdx + 1);
    return above.length ? above : [all[0]];
  }
  const minZ = parseFloat(mode || '0');
  const filtered = all.filter(r => r.z >= minZ);
  return filtered.length ? filtered : [all[0]];
}

function render(j, label) {
  const g = $('grid'); g.replaceChildren();
  const mode = $('cut').value || '0';
  const all = j.results || [];
  const shown = applyCutoff(all, j.baseline, mode);
  shownList = shown;
  const hidden = all.length - shown.length;
  shown.forEach((r, idx) => {
    const c = document.createElement('button'); c.className = 'card'; c.type = 'button';
    const im = document.createElement('img'); im.loading = 'lazy'; im.alt = '';
    im.src = '/thumb?p=' + encodeURIComponent(r.rel);
    const n = document.createElement('div'); n.className = 'n'; n.textContent = r.rel; n.title = r.rel;
    const s = document.createElement('div'); s.className = 's';
    s.textContent = r.score == null ? '' : `${r.score.toFixed(3)}` + (r.z != null ? ` (+${r.z.toFixed(1)}σ)` : '');
    c.append(im, n, s); c.onclick = () => openLb(idx); g.append(c);
  });
  const gpu = info ? ` | budget ${info.budget}` : '';
  const m = j.baseline ? (j.baseline.med ?? j.baseline.mean) : null;
  const d = j.baseline ? (j.baseline.mad ?? j.baseline.std) : null;
  const base = (m != null && d != null) ? ` | med ${m.toFixed(3)}±${d.toFixed(3)}` : '';
  const hidNote = hidden > 0 ? ` (${hidden} below cutoff)` : '';
  status(`${label}: ${shown.length}${hidNote} of ${j.count} | dim ${j.dim}${gpu}${base} | encode ${j.encode_ms} ms, rank ${j.rank_ms} ms`);
}

function textSearch() {
  const q = $('q').value.trim();
  if (activeFile) {
    const lbl = 'image ' + (activeFile.name || 'pasted') + (q ? ` + "${q}"` : '');
    run(lbl, () => fetch('/api/search_image?' + params(q ? {q} : {}),
        {method: 'POST', headers: {'Content-Type': activeFile.type}, body: activeFile}));
    return;
  }
  if (activeRel && q) {
    run(`similar to ${activeRel} + "${q}"`, () => fetch('/api/similar?' + params({p: activeRel, q})));
    return;
  }
  if (!q) { browse(); return; }
  clearImage();
  run(`"${q}"`, () => fetch('/api/search?' + params({q})));
}

function imageSearch(file) {
  if (!file || !file.type.startsWith('image/')) { status('That is not an image.'); return; }
  activeFile = file; activeRel = null;
  $('qimg').src = URL.createObjectURL(file); $('qimg').style.display = 'block';
  const q = $('q').value.trim();
  const lbl = 'image ' + (file.name || 'pasted') + (q ? ` + "${q}"` : '');
  run(lbl, () => fetch('/api/search_image?' + params(q ? {q} : {}),
      {method: 'POST', headers: {'Content-Type': file.type}, body: file}));
}

function openLb(idx) {
  if (!shownList.length) return;
  currentIdx = ((idx % shownList.length) + shownList.length) % shownList.length;
  const r = shownList[currentIdx];
  current = r.rel;
  $('lbimg').src = '/img?p=' + encodeURIComponent(r.rel);
  const sc = r.score == null ? '' : ` — ${r.score.toFixed(3)}` + (r.z != null ? ` (+${r.z.toFixed(1)}σ)` : '');
  $('lbcap').textContent = `[${currentIdx + 1} / ${shownList.length}] ${r.rel}${sc}`;
  $('lb').classList.add('on');
}
function stepLb(delta) {
  if ($('lb').classList.contains('on') && shownList.length) openLb(currentIdx + delta);
}
function closeLb() { $('lb').classList.remove('on'); $('lbimg').removeAttribute('src'); }

$('f').onsubmit = e => { e.preventDefault(); textSearch(); };
$('dim').onchange = $('k').onchange = () => { if (lastRun) run(...lastRun); };
$('cut').onchange = () => { if (lastData) render(lastData, lastLabel); };
$('drop').onclick = () => $('file').click();
$('brw').onclick = browse;
$('qimg').onclick = () => { clearImage(); if ($('q').value.trim()) textSearch(); else browse(); };
$('file').onchange = e => imageSearch(e.target.files[0]);
$('prv').onclick = () => stepLb(-1);
$('nxt').onclick = () => stepLb(1);
$('cls').onclick = closeLb;
$('lb').onclick = e => { if (e.target.id === 'lb') closeLb(); };
$('sim').onclick = () => {
  const rel = current; closeLb();
  activeRel = rel; activeFile = null;
  $('qimg').src = '/thumb?p=' + encodeURIComponent(rel);
  $('qimg').style.display = 'block';
  const q = $('q').value.trim();
  const lbl = 'similar to ' + rel + (q ? ` + "${q}"` : '');
  run(lbl, () => fetch('/api/similar?' + params(q ? {p: rel, q} : {p: rel})));
};
document.addEventListener('keydown', e => {
  if (!$('lb').classList.contains('on')) return;
  if (e.key === 'Escape') closeLb();
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepLb(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepLb(1); }
});
document.addEventListener('dragover', e => { e.preventDefault(); $('drop').classList.add('over'); });
document.addEventListener('dragleave', () => $('drop').classList.remove('over'));
document.addEventListener('drop', e => {
  e.preventDefault(); $('drop').classList.remove('over');
  const f = e.dataTransfer.files[0]; if (f) imageSearch(f);
});
document.addEventListener('paste', e => {
  const it = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (it) imageSearch(it.getAsFile());
});
</script></body></html>
"""
