# main.py — SignBridge v2: INMP441 + Vosk (FR) + mapping n-gram + FastAPI web
import os, json, time, threading, queue, unicodedata
from pathlib import Path
from functools import lru_cache
from collections import deque

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from difflib import get_close_matches

# --- Fuzzy optionnel ----------------------------------------------------------
try:
    from rapidfuzz import process as rf
    HAVE_RF = True
except Exception:
    HAVE_RF = False

# --- Config -------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "vosk-model-small-fr-0.22"
MODEL_DIR = Path(os.getenv("VOSK_MODEL_DIR", str(DEFAULT_MODEL)))

HEADLESS = os.getenv("HEADLESS", "1") == "1"
MIC_DEVICE = os.getenv("MIC_DEVICE")               # ex: "dmic_hw" | "plughw:2,0" | "2"
PORT = int(os.getenv("PORT", "8000"))

VIDEOS = BASE / "videosASL"
VIDEOS.mkdir(exist_ok=True)
EXTS = {".mp4", ".webm", ".avi", ".mov"}

MAPPING_FILE = BASE / "mapping.json"

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_DIR}")

# --- Normalisation + index ----------------------------------------------------
TRANS = str.maketrans('', '', ' _-')
def keyize(s: str) -> str:
    s = unicodedata.normalize('NFD', s.lower().strip())
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').translate(TRANS)

def build_index():
    idx = {}
    for p in VIDEOS.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            idx[keyize(p.stem)] = str(p)
    return idx

INDEX = build_index()
VOCAB = sorted(INDEX)
VOCAB_T = tuple(VOCAB)
if not INDEX:
    print(f"⚠️ Aucune vidéo dans {VIDEOS}")

RAW_MAP = {}
if MAPPING_FILE.exists():
    try:
        RAW_MAP = json.loads(MAPPING_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print("⚠️ mapping.json invalide:", e)

MAP_NORM = {keyize(k): [keyize(v) for v in (vals if isinstance(vals, list) else [vals])]
            for k, vals in RAW_MAP.items()}
MAX_N = max((len(k.split()) for k in MAP_NORM.keys()), default=1)

# --- Web (FastAPI) ------------------------------------------------------------
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

WEB_DIR = BASE / "web"
WEB_DIR.mkdir(exist_ok=True)
app = FastAPI()

app.mount("/static", StaticFiles(directory=str(VIDEOS)), name="static")

LATEST = {"ts": 0.0, "text": "", "videos": []}
LATEST_LOCK = threading.Lock()

def set_current(text: str, video_paths: list[str]) -> None:
    with LATEST_LOCK:
        LATEST["ts"] = time.time()
        LATEST["text"] = text
        LATEST["videos"] = [Path(p).name for p in video_paths]

@app.get("/current")
def current():
    with LATEST_LOCK:
        return dict(LATEST)

@app.get("/manifest")
def manifest():
    return {"items": [{"key": k, "url": f"/static/{Path(p).name}"} for k, p in INDEX.items()]}

@app.get("/")
def root():
    return FileResponse(WEB_DIR / "index.html")

def start_web_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

# --- Lecteur vidéo local (optionnel) ------------------------------------------
class LiveVideoPlayer:
    def __init__(self, name="SignBridge"):
        import cv2
        self.cv2, self.name = cv2, name
        self._last, self._next = None, None
        self._stop, self._switch = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
    def play_now(self, path: str):
        with self._lock:
            self._next = path
            self._switch.set()
    def _loop(self):
        cv2 = self.cv2
        try: cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        except: pass
        cap = None
        while not self._stop.is_set():
            self._switch.wait(0.01)
            if self._switch.is_set():
                self._switch.clear()
                with self._lock: path = self._next
                if cap is not None: cap.release()
                cap = cv2.VideoCapture(path)
                if not cap.isOpened(): print(f"⚠️ Impossible d'ouvrir: {path}"); cap=None
            if cap is not None:
                ok, f = cap.read()
                if ok: self._last = f
                else: cap.release(); cap=None
            if self._last is not None:
                try: cv2.imshow(self.name, self._last)
                except: pass
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'): self._stop.set()
            except: time.sleep(0.01)
        if cap is not None: cap.release()
        try: self.cv2.destroyAllWindows()
        except: pass

# --- STT (Vosk) ---------------------------------------------------------------
print("⏳ Chargement Vosk…", MODEL_DIR)
model = Model(str(MODEL_DIR))

# Taux d'échantillonnage
try:
    default_sr = sd.query_devices(None, 'input')['default_samplerate']
    SR = int(default_sr) if default_sr else 16000
except Exception:
    SR = 16000
rec = KaldiRecognizer(model, SR)
try: rec.SetWords(True)
except: pass

AUDIO_Q = queue.Queue(maxsize=8)

def audio_cb(indata, frames, time_info, status):
    if status: print("⚠️", status)
    try: AUDIO_Q.put_nowait(bytes(indata))
    except queue.Full:
        try: AUDIO_Q.get_nowait()
        except: pass
        try: AUDIO_Q.put_nowait(bytes(indata))
        except: pass

COOLDOWN_NS = int(0.18 * 1e9)
FUZZY, MINLEN = 0.82, 2
last_token = ''
last_time_for = {}

@lru_cache(maxsize=512)
def resolve_exact(tok: str): return INDEX.get(tok)

@lru_cache(maxsize=512)
def resolve_fuzzy(tok: str):
    if HAVE_RF and VOCAB:
        m = rf.extractOne(tok, VOCAB_T, score_cutoff=FUZZY * 100)
        return INDEX.get(m[0]) if m else None
    ms = get_close_matches(tok, VOCAB, n=1, cutoff=FUZZY)
    return INDEX.get(ms[0]) if ms else None

def resolve_word(tok: str): return resolve_exact(tok) or resolve_fuzzy(tok)

# --- N-gram matcher -----------------------------------------------------------
RECENT = deque(maxlen=8)

def _expand_keys_to_paths(keys: list[str]) -> list[str]:
    paths = []
    for k in keys:
        p = resolve_word(k) or INDEX.get(k)
        if p: paths.append(p)
    return paths

def try_match_mapping_from_words(words_norm: list[str]):
    """Retourne (expr_trouvée, [paths]) si une expression du mapping est détectée."""
    if not MAP_NORM: return None
    if not words_norm: return None
    # Fenêtres de taille MAX_N → 1, en partant des plus récentes
    for n in range(min(MAX_N, len(words_norm)), 1-1, -1):
        for i in range(len(words_norm) - n, -1, -1):
            gram = " ".join(words_norm[i:i+n])
            keys = MAP_NORM.get(gram)
            if keys:
                return gram, _expand_keys_to_paths(keys)
    return None

# --- Détection device INMP441 -------------------------------------------------
def _normalize_device(dev):
    if not dev: return None
    try: return int(dev)
    except: return dev

def _pick_dtype(dev):
    n = (str(dev) or "").lower()
    return 'int32' if ('dmic' in n or 'i2s' in n or 'mems' in n) else 'int16'

def _autodetect_device():
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            name = f"{d.get('name','')}".lower()
            if d.get('max_input_channels', 0) > 0 and any(k in name for k in ['dmic','i2s','mems','inmp','ics','mic']):
                return i
        for i, d in enumerate(devs):
            if d.get('max_input_channels', 0) > 0:
                return i
    except Exception:
        pass
    return None
def _choose_samplerate(dev):
    """Retourne un taux d'échantillonnage valide pour ce device d'entrée."""
    # Essaye d'abord le défaut du device, puis une liste de valeurs courantes
    candidates = []
    try:
        dinfo = sd.query_devices(dev, 'input')
        if dinfo and dinfo.get('default_samplerate'):
            candidates.append(int(dinfo['default_samplerate']))
    except Exception:
        pass
    # Ajoute des standards usuels (ordre utile pour I²S)
    candidates += [48000, 44100, 32000, 22050, 16000]

    tried = set()
    for sr in candidates:
        if sr in tried: 
            continue
        tried.add(sr)
        try:
            sd.check_input_settings(device=dev, samplerate=sr, channels=1, dtype=_pick_dtype(dev))
            return sr
        except Exception:
            continue
    raise RuntimeError("Aucun taux d'échantillonnage supporté trouvé pour ce device.")

# --- Boucle principale --------------------------------------------------------
def stt_live(player=None, device=None):
    dev = _normalize_device(device if device is not None else MIC_DEVICE) or _autodetect_device()
    dtype = _pick_dtype(dev)
    SR = _choose_samplerate(dev)  # ← choisit un SR accepté par la carte
    print(f"🎤 LIVE @{SR}Hz (block=1536) device={dev!r} dtype={dtype} RF={'on' if HAVE_RF else 'off'}")

    # Initialiser le recognizer AVEC le même SR que le flux audio
    rec = KaldiRecognizer(model, SR)
    try:
        rec.SetWords(True)
    except:
        pass

    stream = sd.RawInputStream(samplerate=SR, blocksize=1536, dtype=dtype,
                               channels=1, callback=audio_cb, device=dev, latency='low')
    last_parse = 0
    with stream:
        while True:
            # Résultats finaux : on pousse tous les mots dans RECENT
            if rec.AcceptWaveform(AUDIO_Q.get()):
                try:
                    t = json.loads(rec.Result()).get("text", "")
                    if t:
                        print("\n✅ Final:", t)
                        for w in t.split():
                            kw = keyize(w)
                            if len(kw) >= MINLEN:
                                RECENT.append(kw)
                        m = try_match_mapping_from_words(list(RECENT))
                        if m:
                            expr, paths = m
                            if paths:
                                set_current(expr, paths)
                                if player and paths:
                                    player.play_now(paths[0])  # local = première vidéo
                except: pass
            else:
                # Partials pour réactivité
                now = time.monotonic_ns()
                if now - last_parse < 25_000_000:  # 25ms
                    continue
                last_parse = now
                try: part = json.loads(rec.PartialResult()).get("partial", "")
                except: part = ""
                if not part: continue
                words = [keyize(w) for w in part.strip().split() if len(w) >= MINLEN]
                if not words: continue
                # Met à jour le buffer récent sans le figer
                tmp_recent = list(RECENT) + words[-4:]  # limite les derniers pour éviter le bruit
                m = try_match_mapping_from_words(tmp_recent)
                if m:
                    expr, paths = m
                    if paths:
                        set_current(expr, paths)
                        if player:
                            player.play_now(paths[0])

from collections import deque
RECENT = deque(maxlen=8)

def try_mapping(text: str):
    words = [keyize(w) for w in text.split()]
    # on essaye les n-grams de 4→1 mots
    for n in range(4, 0, -1):
        for i in range(0, max(0, len(words) - n + 1)):
            gram = " ".join(words[i:i+n])
            keys = MAP_NORM.get(gram)
            if keys:
                # transforme les clés en chemins selon l'index
                paths = []
                for k in keys:
                    p = INDEX.get(k)
                    if p:
                        paths.append(p)
                if paths:
                    set_current(gram, paths)
                    return True
    return False

# --- Entrée -------------------------------------------------------------------
if __name__ == "__main__":
    print("🎬 Vocabulaire vidéos:", ", ".join(VOCAB) if VOCAB else "(vide)")
    threading.Thread(target=start_web_server, daemon=True).start()
    player = None if HEADLESS else LiveVideoPlayer()
    try:
        stt_live(player=player, device=MIC_DEVICE)
    except KeyboardInterrupt:
        print("\n👋 Arrêt")
