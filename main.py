import json, queue, time, threading, unicodedata, os, asyncio
from typing import Optional
from pathlib import Path
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import cv2
from difflib import get_close_matches

# --- Diffusion web vers les appareils
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  =  Path("/home/pi/PycharmProjects/signbridge/vosk-model-fr-0.22")
VIDEOS_DIR = BASE_DIR / "videosASL"
WEB_DIR    = BASE_DIR / "web"
EXTS = {".mp4", ".webm", ".avi", ".mov"}

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_DIR}")
VIDEOS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)
# ---------- Utils ----------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def keyize(s: str) -> str:
    s = strip_accents(s.lower().strip())
    return s.replace(" ", "").replace("_", "").replace("-", "")

def build_index():
    idx = {}
    for p in VIDEOS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            idx[keyize(p.stem)] = str(p)
    return idx

INDEX = build_index()
VOCAB = sorted(INDEX.keys())

if not INDEX:
    print(f"⚠️ Aucune vidéo trouvée dans {VIDEOS_DIR}. Ajoute p.ex. {VIDEOS_DIR/'bonjour.mp4'}")

# ---------- Lecteur vidéo robuste ----------
class LiveVideoPlayer:
    """Joue toujours la dernière vidéo demandée; garde la fenêtre ouverte et la dernière frame affichée."""
    def __init__(self, window_name="SignBridge"):
        self.window = window_name
        self._lock = threading.Lock()
        self._next_path = None
        self._switch = threading.Event()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._last_frame = None
        self._t.start()

    def play_now(self, path: str):
        with self._lock:
            self._next_path = path
            self._switch.set()

    def _ensure_window(self):
        # recrée la fenêtre si elle est perdue
        try:
            cv2.getWindowProperty(self.window, 0)
        except Exception:
            pass
        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        except Exception:
            pass

    def _loop(self):
        self._ensure_window()
        cap = None
        while not self._stop.is_set():
            if self._switch.is_set():
                self._switch.clear()
                with self._lock:
                    path = self._next_path
                if cap is not None:
                    cap.release()
                    cap = None
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print(f"⚠️ Impossible d'ouvrir la vidéo: {path}")
                    cap = None

            frame_shown = False
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    # fin → rester sur la dernière frame
                    cap.release()
                    cap = None
                else:
                    self._last_frame = frame
                    frame_shown = True

            # Afficher soit la frame courante, soit la dernière frame connue
            if frame_shown and self._last_frame is not None:
                try:
                    cv2.imshow(self.window, self._last_frame)
                except Exception as e:
                    # si la fenêtre a disparu, on la recrée
                    self._ensure_window()
            elif self._last_frame is not None:
                # maintenir l'UI visible (image figée)
                try:
                    cv2.imshow(self.window, self._last_frame)
                except Exception:
                    self._ensure_window()

            # UI réactive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop.set()
                break

            if cap is None and not self._switch.is_set():
                time.sleep(0.005)

        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def close(self):
        self._stop.set()
        self._t.join(timeout=1)

app = FastAPI()
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
app.mount("/static", StaticFiles(directory=str(VIDEOS_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="site")
CLIENTS = set()
WS_LOOP = None
LATEST = {"ts": 0.0, "text": "", "videos": []}
LATEST_LOCK = threading.Lock()
@app.on_event("startup")
async def _startup():
    global WS_LOOP
    WS_LOOP = asyncio.get_running_loop()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            # on ignore les messages entrants
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)

@app.get("/manifest")
def manifest():
    items = []
    for k, p in INDEX.items():
        rel = Path(p)
        try:
            rel = rel.relative_to(VIDEOS_DIR)
        except Exception:
            rel = Path(p).name
        url = "/videos/" + str(rel).replace(os.sep, "/")
        items.append({"key": k, "url": url})
    return {"items": items}

async def _ws_broadcast(msg: dict):
    dead = []
    for w in list(CLIENTS):
        try:
            await w.send_json(msg)
        except Exception:
            dead.append(w)
    for w in dead:
        CLIENTS.discard(w)

def _to_url(path: str) -> str:
    p = Path(path)
    try:
        rel = p.relative_to(VIDEOS_DIR)
    except Exception:
        rel = p.name
    return "/videos/" + str(rel).replace(os.sep, "/")

def notify_play(path: str, text: Optional[str] = None):
    if not WS_LOOP:
        return
    url = _to_url(path)
    asyncio.run_coroutine_threadsafe(_ws_broadcast({"type": "play", "url": url, "ts": time.time()}), WS_LOOP)
    if text:
        asyncio.run_coroutine_threadsafe(_ws_broadcast({"type": "text", "text": text}), WS_LOOP)

def set_current(text: str, video_paths: list[str]) -> None:
    """Publie la dernière séquence pour /current (on n’envoie que les noms de fichiers)."""
    with LATEST_LOCK:
        LATEST["ts"] = time.time()
        LATEST["text"] = text
        LATEST["videos"] = [Path(p).name for p in video_paths]

@app.get("/current")
def current():
    # La page web lit ceci périodiquement
    with LATEST_LOCK:
        return dict(LATEST)

def start_web_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

# ---------- Vosk (latence basse + grammaire + fallback) ----------
print("⏳ Chargement du modèle Vosk…")
model = Model(str(MODEL_DIR))

try:
    default_sr = sd.query_devices(None, 'input')['default_samplerate']
    SAMPLERATE = int(default_sr) if default_sr else 16000
except Exception:
    SAMPLERATE = 16000

# Grammaire contrainte (= nos mots) pour accélérer; sinon décodage libre
grammar = json.dumps(VOCAB) if VOCAB else None
rec = KaldiRecognizer(model, SAMPLERATE, grammar) if grammar else KaldiRecognizer(model, SAMPLERATE)
try:
    rec.SetWords(True)  # pas obligatoire, mais utile pour stabiliser certains modèles
except Exception:
    pass

audio_q = queue.Queue()

# Anti-spam & stabilité
last_token = ""
last_time_for = {}
COOLDOWN_SEC = 0.18          # ~5 déclenchements max/s pour un même mot
FUZZY_CUTOFF = 0.82          # tolérance pour mapping approximatif (0..1)
MIN_TOKEN_LEN = 2            # ignorer les tokens trop courts (bruit)

def resolve_token(tok: str) -> str | None:
    """Retourne le chemin vidéo pour un tok (exact ou fuzzy)."""
    if tok in INDEX:
        return INDEX[tok]
    # fallback fuzzy s'il n'y a pas de match exact
    matches = get_close_matches(tok, VOCAB, n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        return INDEX[matches[0]]
    return None

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_q.put(bytes(indata))

def stt_live(player: Optional[LiveVideoPlayer], device=None):
    print(f"🎤 LIVE @ {SAMPLERATE} Hz — latence basse (blocksize 1536).")
    phrase_buf = []
    last_input_t = 0.0

    def commit_phrase():
        nonlocal phrase_buf
        if not phrase_buf:
            return
        label = " ".join(phrase_buf)
        seq = []
        for t in phrase_buf:
            p = resolve_token(t)
            if p:
                seq.append(p)
        for p in seq:
            notify_play(p, label)
        if seq and player is not None:
            player.play_now(seq[0])
        phrase_buf = []
    with sd.RawInputStream(samplerate=SAMPLERATE,
                           blocksize=1536,               # compromis vitesse/stabilité
                           dtype='int16',
                           channels=1,
                           callback=audio_callback,
                           device=device,
                           latency='low'):
        while True:
            data = audio_q.get()
            if rec.AcceptWaveform(data):
                # On n’en a pas besoin pour déclencher rapidement,
                # mais ça peut afficher le segment reconnu.
                try:
                    final_text = json.loads(rec.Result()).get("text", "")
                    if final_text:
                        print(f"\n✅ Final: {final_text}")
                except Exception:
                    pass
            else:
                try:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                except Exception:
                    partial = ""

                if partial:
                    tokens = partial.strip().lower().split()
                    if not tokens:
                        continue
                    tok_raw = tokens[-1]
                    if len(tok_raw) < MIN_TOKEN_LEN:
                        continue
                    tok = keyize(tok_raw)

                    global last_token
                    if tok and tok != last_token:
                        now = time.time()
                        if now - last_time_for.get(tok, 0) >= COOLDOWN_SEC:
                            last_time_for[tok] = now
                            path = resolve_token(tok)
                            if path:
                                print(f"\r▶️ {tok_raw} ", end="", flush=True)
                                player.play_now(path)
                                set_current(tok_raw, [path])
                        last_token = tok

# ---------- Main ----------
if __name__ == "__main__":
    if VOCAB:
        print("🎬 Vocabulaire:", ", ".join(VOCAB))
    else:
        print("ℹ️ Vocabulaire vide (ajoute des vidéos).")
    # démarrer le serveur web (pour téléphone/PC)
    web_t = threading.Thread(target=start_web_server, daemon=True)
    web_t.start()

    player = LiveVideoPlayer("SignBridge")
    try:
        stt_live(player)
    except KeyboardInterrupt:
        print("\n👋 Arrêt")
    finally:
        player.close()
