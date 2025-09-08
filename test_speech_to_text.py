import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

model_path = r"C:\Users\HP\Documents\BD\PythonProject\model\vosk-model-small-fr-0.22"
model = Model(model_path)
samplerate = 16000
q = queue.Queue()
rec = KaldiRecognizer(model, samplerate)

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("🎤 Parle dans le micro... (Ctrl+C pour arrêter)")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print("✅ Texte reconnu :", result["text"])
        else:
            partial = json.loads(rec.PartialResult())
            if partial["partial"]:
                print("... ", partial["partial"])
