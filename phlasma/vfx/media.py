"""ffmpeg in/out, audio analysis, image loading, text, and the diagnostic sheets agents look at."""
import json, os, shutil, subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont



def _find_ffmpeg():
    """System ffmpeg if on PATH, else the binary bundled with imageio-ffmpeg (Windows services/tasks often lack PATH)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG = _find_ffmpeg()
SR = 44100
BRAND = os.environ.get("PHLASMA_BRAND", "")  # a folder holding fonts/ (Newsreader, IBM Plex Mono); else PIL default
_here = os.path.dirname(__file__)
FONT_FILES = {"serif": "Newsreader-300-opsz21.ttf", "serif_light": "Newsreader-200-opsz21.ttf",
              "mono": "IBMPlexMono-Regular.ttf", "mono_bold": "IBMPlexMono-Bold.ttf"}
ACCENT = (0x1D, 0x84, 0x7E)
INK = (64, 64, 64)


def font(kind, px):
    for d in (os.path.join(_here, "fonts"), os.path.join(BRAND, "fonts")):
        f = os.path.join(d, FONT_FILES[kind])
        if os.path.exists(f):
            return ImageFont.truetype(f, max(int(px), 4))
    return ImageFont.load_default(size=max(int(px), 4))


# --------------------------------------------------------------------------------------------- audio
def decode_audio(path):
    p = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(f"ffmpeg could not read audio {path}: {p.stderr.decode()[-400:]}")
    return np.frombuffer(p.stdout, np.float32)


def analyse(samples, fps, n_frames, offset=0.0, bands=((20, 150), (150, 2000), (2000, 12000)), normalize=True, gain=1.0):
    """Per-frame features with a trailing 2048-sample window ending at each frame's time."""
    N = 2048
    win = np.hanning(N).astype(np.float32)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    masks = [(freqs >= a) & (freqs < b) for a, b in bands]
    pad = np.concatenate([np.zeros(N, np.float32), samples * gain])
    feats = np.zeros((n_frames, 5), np.float32)
    prev = None
    for f in range(n_frames):
        end = int((f / fps + offset) * SR) + N
        seg = pad[max(end - N, 0):end]
        if len(seg) < N:
            seg = np.pad(seg, (0, N - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win)) / N
        rms = float(np.sqrt(np.mean(seg ** 2)))
        b = [float(mag[m].mean()) if m.any() else 0.0 for m in masks]
        lm = np.log1p(mag * 1000)
        flux = 0.0 if prev is None else float(np.maximum(lm - prev, 0).sum())
        prev = lm
        feats[f] = [rms, *b, flux]
    if normalize and n_frames:
        ref = np.percentile(feats, 98, axis=0)
        feats = feats / np.where(ref > 1e-9, ref, 1.0)
    return feats  # columns: rms low mid high onset


# --------------------------------------------------------------------------------------------- images
def fit_image(img, w, h, fit):
    if fit == "stretch":
        return img.resize((w, h), Image.LANCZOS)
    iw, ih = img.size
    s = max(w / iw, h / ih) if fit == "fill" else min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    r = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return out


def load_image(path, w, h, fit):
    img = Image.open(path).convert("RGBA")
    return np.asarray(fit_image(img, w, h, fit), np.float32) / 255.0


class MovieReader:
    def __init__(self, path, w, h, fps, fit, offset, loop):
        self.args = (path, w, h, fps, fit, offset, loop)
        self.proc, self.pos = None, None
        self.w, self.h = w, h

    def _open(self, t):
        path, w, h, fps, fit, offset, loop = self.args
        if self.proc:
            self.proc.kill()
        if fit == "stretch":
            vf = f"scale={w}:{h}"
        elif fit == "fill":
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        else:
            vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        cmd = [FFMPEG, "-v", "error"]
        if loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-ss", f"{max(t, 0):.4f}", "-i", path, "-vf", f"{vf},fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last = np.zeros((h, w, 4), np.float32)

    def frame(self, f):
        path, w, h, fps, fit, offset, loop = self.args
        if self.pos is None or f != self.pos:
            self._open(offset + f / fps)
            self.pos = f
        n = w * h * 4
        buf = self.proc.stdout.read(n)
        self.pos = f + 1
        if len(buf) == n:
            self.last = np.frombuffer(buf, np.uint8).reshape(h, w, 4).astype(np.float32) / 255.0
        return self.last

    def close(self):
        if self.proc:
            self.proc.kill()


def render_text(text, kind, size, pos, align, color, w, h):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(kind, size * h * 1.4)
    anchor = {"left": "ls", "center": "ms", "right": "rs"}[align]
    col = tuple(int(max(0, min(1, c)) * 255) for c in color)
    d.text((pos[0] * w, pos[1] * h), text, font=f, fill=col, anchor=anchor)
    return np.asarray(img, np.float32) / 255.0


# --------------------------------------------------------------------------------------------- outputs
def to8(a):
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_png(a, path, alpha=False):
    im = to8(a if alpha else a[..., :3])
    Image.fromarray(im).save(path)


class VideoWriter:
    def __init__(self, path, w, h, fps, crf=18):
        self.path = path
        self.tmp = path + ".noaudio.mp4"
        self.proc = subprocess.Popen(
            [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", self.tmp], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, a):
        self.proc.stdin.write(to8(a[..., :3]).tobytes())

    def close(self, audio=None, audio_offset=0.0):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode()
        if self.proc.wait():
            raise RuntimeError(f"ffmpeg encode failed: {err[-400:]}")
        if audio:
            p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", self.tmp, "-ss", f"{audio_offset:.4f}", "-i", audio,
                                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                "-shortest", self.path], capture_output=True)
            if p.returncode:
                raise RuntimeError(f"ffmpeg mux failed: {p.stderr.decode()[-400:]}")
            os.remove(self.tmp)
        else:
            os.replace(self.tmp, self.path)


def _label(d, xy, text, f, chip=True):
    if chip:
        l, t, r, b = d.textbbox(xy, text, font=f)
        d.rectangle((l - 5, t - 3, r + 5, b + 3), fill=ACCENT)
        d.text(xy, text, font=f, fill=(255, 255, 255))
    else:
        d.text(xy, text, font=f, fill=INK)


def contact_sheet(frames, labels, path, cols=4, cell_w=360):
    """frames: list of HxWx4 arrays. White page, brand chips, mono captions."""
    if not frames:
        return
    h0, w0 = frames[0].shape[:2]
    cw, ch = cell_w, int(cell_w * h0 / w0)
    rows = (len(frames) + cols - 1) // cols
    g, cap = 12, 26
    W, H = cols * cw + (cols + 1) * g, rows * (ch + cap) + (rows + 1) * g
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = font("mono", 12)
    for i, (a, lab) in enumerate(zip(frames, labels)):
        x, y = g + (i % cols) * (cw + g), g + (i // cols) * (ch + cap + g)
        sheet.paste(Image.fromarray(to8(a[..., :3])).resize((cw, ch), Image.LANCZOS), (x, y))
        _label(d, (x, y + ch + 7), lab, f, chip=False)
    sheet.save(path)


def node_sheet(items, path, cell_w=300):
    """items: list of dicts {id, op, img, ms, stats, issue}. One cell per node in evaluation order."""
    if not items:
        return
    cols = min(5, len(items))
    h0, w0 = items[0]["img"].shape[:2]
    cw, ch = cell_w, int(cell_w * h0 / w0)
    g, cap = 12, 44
    rows = (len(items) + cols - 1) // cols
    W, H = cols * cw + (cols + 1) * g, rows * (ch + cap) + (rows + 1) * g
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    fm, fs = font("mono", 12), font("mono", 11)
    for i, it in enumerate(items):
        x, y = g + (i % cols) * (cw + g), g + (i // cols) * (ch + cap + g)
        a = it["img"]
        # checker under alpha so transparent nodes read as transparent
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        chk = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 0.92, 0.8)
        comp = a[..., :3] * np.clip(a[..., 3:4], 0, 1) + chk * (1 - np.clip(a[..., 3:4], 0, 1))
        sheet.paste(Image.fromarray(to8(comp)).resize((cw, ch), Image.LANCZOS), (x, y))
        _label(d, (x + 4, y + ch + 8), it["id"], fm, chip=True)
        s = it.get("stats", {})
        cap2 = f'{it["op"]}  {it.get("ms", 0):.1f} ms  mean {s.get("mean", [0])[0]:.2f}'
        if it.get("issue"):
            cap2 = "! " + it["issue"]
        _label(d, (x, y + ch + 27), cap2, fs, chip=False)
    sheet.save(path)


def stats(a):
    finite = np.isfinite(a)
    nan = int((~finite).any(axis=-1).sum())
    b = np.where(finite, a, 0)
    return {"mean": [round(float(x), 4) for x in b.reshape(-1, 4).mean(0)],
            "min": round(float(b[..., :3].min()), 4), "max": round(float(b[..., :3].max()), 4),
            "nan_px": nan, "over_1": round(float((b[..., :3] > 1.0).any(-1).mean()), 4)}


def dump(o, path):
    with open(path, "w") as f:
        json.dump(o, f, indent=1, default=str)
