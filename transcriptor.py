"""Transcriptor de llamadas en vivo (Espanol / Ingles).
Escucha tu microfono ("Yo") y el audio que suena en la PC ("Ellos"),
transcribe con Whisper localmente y guarda todo en un .txt mientras hablas.
Requisitos:  pip install -r requirements.txt
"""
import datetime
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pyaudiowpatch as pyaudio


def _activar_dlls_nvidia():
    """Permite usar la GPU NVIDIA con las librerias pip nvidia-cublas / nvidia-cudnn."""
    try:
        import nvidia.cublas, nvidia.cudnn
    except ImportError:
        return
    for mod in (nvidia.cublas, nvidia.cudnn):
        for p in mod.__path__:
            b = os.path.join(p, "bin")
            if os.path.isdir(b):
                os.add_dll_directory(b)
                os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]


_activar_dlls_nvidia()
from faster_whisper import WhisperModel  # noqa: E402

SR = 16000            # Whisper trabaja a 16 kHz
BLOQUE = 0.25         # segundos por lectura de audio
SILENCIO_FIN = 0.6    # segundos de silencio para cerrar una frase
MAX_FRASE = 10        # segundos maximos por fragmento
UMBRAL = 0.006        # volumen minimo para considerar que hay voz

CARPETA = os.path.join(os.path.expanduser("~"), "Documents", "Transcripciones")

NOMBRES = {
    "es": "Español", "en": "Inglés", "fr": "Francés", "de": "Alemán", "it": "Italiano",
    "pt": "Portugués", "ru": "Ruso", "zh": "Chino", "ja": "Japonés", "ko": "Coreano",
    "ar": "Árabe", "tr": "Turco", "nl": "Holandés", "pl": "Polaco", "hi": "Hindi",
    "uk": "Ucraniano", "sv": "Sueco", "el": "Griego", "he": "Hebreo", "id": "Indonesio",
}
IDIOMAS = {"Automático (Español / Inglés)": None, "Automático (cualquier idioma)": "auto"}
IDIOMAS.update({n: c for c, n in NOMBRES.items()})
DESTINOS = {"No traducir": None}
DESTINOS.update({n: c for c, n in NOMBRES.items()})
MODELOS = {
    "small - recomendado": "small",
    "base - mas rapido": "base",
    "medium - mas preciso": "medium",
    "turbo - el mejor (necesita GPU)": "large-v3-turbo",
}


class _TokSP:
    def __init__(self, sp):
        self.sp = sp

    def encode(self, texto):
        return self.sp.encode(texto, out_type=str)

    def decode(self, tokens):
        return " ".join(self.sp.decode_pieces(tokens).replace("▁", " ").split())


class _TokBPE:
    """Tokenizador de los paquetes nuevos de Argos (Moses + BPE de subword-nmt)."""

    def __init__(self, ruta, origen, destino):
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
        from subword_nmt.apply_bpe import BPE
        self.norm = MosesPunctNormalizer(origen)
        self.tok = MosesTokenizer(origen)
        self.detok = MosesDetokenizer(destino)
        with open(ruta, encoding="utf-8") as f:
            self.bpe = BPE(f)

    def encode(self, texto):
        palabras = " ".join(self.tok.tokenize(self.norm.normalize(texto))).strip().split(" ")
        return self.bpe.segment_tokens(palabras)

    def decode(self, tokens):
        return self.detok.detokenize(" ".join(tokens).replace("@@ ", "").split(" "))


class Traductor:
    """Traduccion entre muchos idiomas sin internet (modelos de Argos Translate).
    Descarga cada par la primera vez; si no hay par directo, pasa por ingles."""

    INDICE = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
    RAIZ = os.path.join(os.path.expanduser("~"), ".cache", "traductor")

    def __init__(self):
        self._indice = None
        self._pares = {}
        self._lock = threading.Lock()

    @staticmethod
    def _pedir(url):
        import urllib.request
        return urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    def _cargar_indice(self):
        if self._indice is None:
            import json, urllib.request
            with urllib.request.urlopen(self._pedir(self.INDICE), timeout=30) as r:
                self._indice = json.load(r)
        return self._indice

    def _carpeta(self, a, b):
        return os.path.join(self.RAIZ, f"{a}_{b}")

    @staticmethod
    def _buscar_modelo(carpeta):
        for raiz, _, archivos in os.walk(carpeta):
            if ("sentencepiece.model" in archivos or "bpe.model" in archivos)                     and os.path.isdir(os.path.join(raiz, "model")):
                return raiz
        return None

    def existe(self, a, b):
        return (self._buscar_modelo(self._carpeta(a, b)) is not None
                or any(p["from_code"] == a and p["to_code"] == b for p in self._cargar_indice()))

    def _descargar(self, a, b, aviso):
        import io, urllib.request, zipfile
        pkg = next(p for p in self._cargar_indice() if p["from_code"] == a and p["to_code"] == b)
        url = next(u for u in pkg["links"] if u.startswith("http"))
        aviso(f"Cargando traductor {a} → {b} (la primera vez se descarga, puede tardar)...")
        with urllib.request.urlopen(self._pedir(url), timeout=300) as r:
            datos = r.read()
        os.makedirs(self._carpeta(a, b), exist_ok=True)
        zipfile.ZipFile(io.BytesIO(datos)).extractall(self._carpeta(a, b))

    def _par(self, a, b, aviso):
        if (a, b) not in self._pares:
            import ctranslate2
            import sentencepiece as spm
            base = self._buscar_modelo(self._carpeta(a, b))
            if base is None:
                self._descargar(a, b, aviso)
                base = self._buscar_modelo(self._carpeta(a, b))
            ruta_sp = os.path.join(base, "sentencepiece.model")
            if os.path.exists(ruta_sp):
                tok = _TokSP(spm.SentencePieceProcessor(model_file=ruta_sp))
            else:   # paquetes nuevos: BPE + tokenizador Moses
                tok = _TokBPE(os.path.join(base, "bpe.model"), a, b)
            self._pares[(a, b)] = (
                tok, ctranslate2.Translator(os.path.join(base, "model"), device="cpu", compute_type="int8"))
        return self._pares[(a, b)]

    def _paso(self, texto, a, b, aviso):
        import re
        sp, tr = self._par(a, b, aviso)
        frases = [f for f in re.split(r"(?<=[.!?。！？])\s*", texto) if f.strip()] or [texto]
        res = tr.translate_batch([sp.encode(f) for f in frases], beam_size=4)
        return " ".join(sp.decode(r.hypotheses[0]) for r in res)

    def traducir(self, texto, origen, destino, aviso=lambda t: None):
        """Devuelve la traduccion o None si no hay ruta entre los idiomas."""
        if origen == destino:
            return None
        with self._lock:
            if self.existe(origen, destino):
                return self._paso(texto, origen, destino, aviso)
            if origen != "en" and destino != "en" and self.existe(origen, "en") and self.existe("en", destino):
                return self._paso(self._paso(texto, origen, "en", aviso), "en", destino, aviso)
        return None


# Frases que Whisper "inventa" cuando escucha musica o silencio
ALUCINACIONES_EXACTAS = {
    "musica", "música", "music", "gracias", "gracias por ver", "gracias por ver el video",
    "suscribete", "suscríbete", "thanks for watching", "thank you for watching",
    "thank you", "subscribe", "aplausos", "applause", "risas",
}
ALUCINACIONES_CONTIENE = ("amara.org", "subtítulos por", "subtitulos por", "subtitulado por",
                          "subtítulos realizados", "subtitles by")


def es_alucinacion(t, estricto):
    t = t.strip().lower()
    limpio = t.strip(" .,!?¿¡♪♫-[]()*")
    return (not limpio or any(a in t for a in ALUCINACIONES_CONTIENE)
            or (estricto and limpio in ALUCINACIONES_EXACTAS))


def cargar_modelo(nombre):
    """Intenta usar la GPU NVIDIA; si no se puede, usa el procesador."""
    try:
        m = WhisperModel(nombre, device="cuda", compute_type="int8_float16")
        list(m.transcribe(np.zeros(SR, np.float32), language="es")[0])  # prueba real en GPU
        return m, "GPU"
    except Exception:
        return WhisperModel(nombre, device="cpu", compute_type="int8"), "CPU"


def dispositivos(pa):
    """Devuelve (microfono, loopback de los altavoces) por defecto en WASAPI."""
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    mic = pa.get_device_info_by_index(wasapi["defaultInputDevice"])
    altavoz = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    loop = None
    for d in pa.get_loopback_device_info_generator():
        if altavoz["name"] in d["name"]:
            loop = d
            break
    return mic, loop


class Capturador(threading.Thread):
    """Lee audio de un dispositivo y corta frases usando los silencios."""

    def __init__(self, quien, pa, dev, cola, activo, on_error):
        super().__init__(daemon=True)
        self.quien, self.pa, self.dev, self.cola = quien, pa, dev, cola
        self.activo, self.on_error = activo, on_error

    def _a_16k(self, x, rate):
        if rate == SR:
            return x
        if rate % SR == 0:  # p.ej. 48000 -> promedio de cada 3 muestras
            k = rate // SR
            return x[: len(x) // k * k].reshape(-1, k).mean(axis=1)
        m = int(len(x) * SR / rate)
        return np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype(np.float32)

    def run(self):
        rate = int(self.dev["defaultSampleRate"])
        ch = max(1, int(self.dev["maxInputChannels"]))
        k = rate // SR if rate % SR == 0 else 1
        crudo = queue.Queue()

        def callback(in_data, frame_count, time_info, status):
            crudo.put(in_data)
            return (None, pyaudio.paContinue)

        try:
            stream = self.pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                                  input_device_index=self.dev["index"],
                                  frames_per_buffer=int(rate * 0.1), stream_callback=callback)
        except Exception as e:
            self.on_error(f"No se pudo abrir '{self.dev['name']}' ({self.quien}):\n{e}")
            return

        n = int(SR * BLOQUE)
        resto_crudo = np.zeros(0, np.float32)
        pendiente = np.zeros(0, np.float32)
        buf, previo, silencio, inicio = [], None, 0.0, None
        try:
            while self.activo.is_set():
                try:
                    d = crudo.get(timeout=BLOQUE)
                    x = np.frombuffer(d, np.float32).reshape(-1, ch).mean(axis=1)
                    x = np.concatenate([resto_crudo, x])
                    usar = len(x) // k * k
                    resto_crudo = x[usar:]
                    x = self._a_16k(x[:usar], rate)
                except queue.Empty:
                    x = np.zeros(n, np.float32)  # el loopback no entrega nada si no suena audio
                pendiente = np.concatenate([pendiente, x])

                while len(pendiente) >= n:
                    b, pendiente = pendiente[:n], pendiente[n:]
                    voz = np.sqrt(np.mean(b ** 2)) > UMBRAL
                    if voz and not buf:
                        inicio = datetime.datetime.now() - datetime.timedelta(seconds=BLOQUE)
                        if previo is not None:
                            buf.append(previo)  # no cortar la primera silaba
                    if buf or voz:
                        buf.append(b)
                        silencio = 0.0 if voz else silencio + BLOQUE
                    previo = b
                    if buf and (silencio >= SILENCIO_FIN or len(buf) * BLOQUE >= MAX_FRASE):
                        self.cola.put((inicio, self.quien, np.concatenate(buf)))
                        buf, silencio = [], 0.0
            if buf:
                self.cola.put((inicio, self.quien, np.concatenate(buf)))
        except Exception as e:
            self.on_error(f"{self.quien}: {e}")
        finally:
            stream.stop_stream()
            stream.close()


# Paleta inspirada en Tole Tole (leopardo de las nieves): cielo turquesa, pasto y pelaje gris
C_FONDO = "#e6f1f0"        # fondo general (turquesa muy claro)
C_TARJETA = "#ffffff"
C_BORDE = "#cfe1df"
C_TEXTO = "#1d2a2c"
C_SUAVE = "#5f7577"
C_CIELO = "#3f9ea3"        # turquesa del cielo
C_CIELO_OSC = "#2e8388"
C_PASTO = "#5fa33f"        # verde del pasto
C_PELO = "#eef2f2"         # pelaje claro
C_TRAD = "#8a5a12"
C_TRAD_BG = "#fff3d1"
ESTADOS = {"idle": "#9fb3b5", "busy": "#f0a52b", "live": "#4caf50", "error": "#e5484d"}
FUENTE = "Segoe UI"


class Asistente:
    """IA local con Ollama: responde preguntas sobre la transcripcion. Gratis y sin internet."""

    URL = "http://127.0.0.1:11434"
    MODELO = "qwen2.5:3b"
    MAX_CHARS = 14000   # ~4-5k tokens de transcripcion; si es mas larga se usa la parte reciente
    EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")

    def _http(self, ruta, datos=None, timeout=10):
        import json, urllib.request
        req = urllib.request.Request(
            self.URL + ruta, data=None if datos is None else json.dumps(datos).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    proc = None   # proceso de Ollama que arranco este programa (no toca uno que ya estaba abierto)

    def cerrar(self):
        """Apaga Ollama si lo abrio este programa, para no dejarlo en segundo plano."""
        if self.proc is not None and self.proc.poll() is None:
            import subprocess
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           creationflags=0x08000000, capture_output=True)
        self.proc = None

    def _vivo(self):
        try:
            self._http("/api/tags", timeout=2).close()
            return True
        except Exception:
            return False

    def preparar(self, aviso):
        """Arranca Ollama si hace falta y descarga el modelo la primera vez."""
        import json, subprocess, time
        if not self._vivo():
            exe = self.EXE if os.path.exists(self.EXE) else "ollama"
            try:
                self.proc = subprocess.Popen([exe, "serve"], creationflags=0x08000000,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                raise RuntimeError("Ollama no esta instalado. Instalalo desde https://ollama.com")
            for _ in range(30):
                if self._vivo():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("No se pudo iniciar Ollama.")
        with self._http("/api/tags") as r:
            nombres = [m["name"] for m in json.load(r).get("models", [])]
        if self.MODELO not in nombres:
            aviso(f"Descargando el modelo de IA '{self.MODELO}' (~2 GB, solo la primera vez)...")
            with self._http("/api/pull", {"name": self.MODELO}, timeout=3600) as r:
                for linea in r:
                    d = json.loads(linea)
                    if d.get("total") and d.get("completed"):
                        aviso(f"Descargando modelo de IA... {d['completed'] * 100 // d['total']}%")

    def preguntar(self, transcripcion, pregunta, historial, on_texto):
        import json
        if len(transcripcion) > self.MAX_CHARS:
            transcripcion = "(...inicio omitido...)\n" + transcripcion[-self.MAX_CHARS:]
        sistema = (
            "Eres Tole Tole, un asistente amable que ayuda con la transcripcion de una llamada. "
            "Siempre respondes en espanol con frases completas y breves, nunca con un solo simbolo.\n\n"
            "Como responder segun el tipo de mensaje:\n"
            "1) Saludo o charla casual (hola, gracias, como estas): contesta con naturalidad y ofrece ayuda. "
            "Ejemplo: usuario 'hola' -> 'Hola, soy Tole Tole. Preguntame lo que quieras sobre la llamada.'\n"
            "2) Pregunta sobre la llamada: responde usando SOLO la transcripcion ('Yo' es el usuario y "
            "'Ellos' la otra persona). Ejemplo: 'En la llamada no se menciona eso.' si el dato no esta.\n"
            "3) Algo que no tiene que ver con la llamada: explica amablemente que solo ayudas con lo que "
            "se dijo en ella.\n\n"
            "TRANSCRIPCION:\n" + (transcripcion or "(todavia no hay nada transcrito)"))
        msgs = [{"role": "system", "content": sistema}] + historial[-6:] + [
            {"role": "user", "content": pregunta}]
        datos = {"model": self.MODELO, "messages": msgs, "stream": True,
                 "options": {"num_ctx": 8192, "temperature": 0.3}}
        total = ""
        with self._http("/api/chat", datos, timeout=600) as r:
            for linea in r:
                trozo = json.loads(linea).get("message", {}).get("content", "")
                if trozo:
                    total += trozo
                    on_texto(trozo)
        return total


class App:
    def __init__(self, root):
        self.root = root
        root.title("Tole Tole · Transcriptor de Llamadas")
        root.geometry("1280x720")
        root.minsize(720, 540)
        root.configure(bg=C_FONDO)

        self.modelo = None
        self.modelo_nombre = None
        self.pa = None
        self.dispositivo = "CPU"
        self.traductor = None
        self.traductor_activo = False
        self.activo = threading.Event()
        self.cola = queue.Queue()
        self.archivo = None
        self.n_mensajes = 0
        self.historial_tx = []    # (hora, quien, texto) para la IA
        self.ia = Asistente()
        self.ia_listo = False
        self.ia_ocupada = False
        self.ia_chat = []
        self.etiquetas = []   # labels de mensajes (para ajustar el ajuste de linea al redimensionar)

        self._estilos()

        # ---- encabezado ----
        cab = tk.Frame(root, bg=C_CIELO)
        cab.pack(fill="x")
        try:
            ruta_av = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tole.png")
            self.avatar = tk.PhotoImage(file=ruta_av)
            root.iconphoto(True, self.avatar)
            av = self.avatar.subsample(2, 2)   # 64 px
            self.avatar_chico = av
            tk.Label(cab, image=av, bg=C_CIELO).pack(side="left", padx=(20, 12), pady=8)
        except Exception:
            tk.Label(cab, text="🐾", bg=C_CIELO, fg="white", font=(FUENTE, 24)).pack(side="left", padx=(20, 8), pady=10)
        titulos = tk.Frame(cab, bg=C_CIELO)
        titulos.pack(side="left", pady=10)
        tk.Label(titulos, text="Transcriptor de Llamadas", bg=C_CIELO, fg="white",
                 font=(FUENTE + " Semibold", 16)).pack(anchor="w")
        tk.Label(titulos, text="Tole Tole escucha por ti", bg=C_CIELO, fg="#d9f2f1",
                 font=(FUENTE, 9)).pack(anchor="w")
        tk.Label(cab, text="Español / Inglés · en vivo · sin internet", bg=C_CIELO, fg="#d9f2f1",
                 font=(FUENTE, 10)).pack(side="right", padx=20)
        tk.Frame(root, bg=C_PASTO, height=4).pack(fill="x")   # franja de pasto

        # ---- panel de configuracion ----
        panel = tk.Frame(root, bg=C_TARJETA, highlightthickness=1, highlightbackground=C_BORDE)
        panel.pack(fill="x", padx=16, pady=(14, 8))
        cfg = tk.Frame(panel, bg=C_TARJETA)
        cfg.pack(fill="x", padx=16, pady=12)
        cfg.columnconfigure(1, weight=1)
        cfg.columnconfigure(3, weight=1)

        ttk.Label(cfg, text="Idioma", style="Campo.TLabel").grid(row=0, column=0, sticky="w")
        self.idioma = ttk.Combobox(cfg, values=list(IDIOMAS), state="readonly", width=28)
        self.idioma.current(0)
        self.idioma.grid(row=0, column=1, sticky="ew", padx=(8, 20))

        ttk.Label(cfg, text="Modelo", style="Campo.TLabel").grid(row=0, column=2, sticky="w")
        self.modelo_cb = ttk.Combobox(cfg, values=list(MODELOS), state="readonly", width=28)
        self.modelo_cb.current(0)
        self.modelo_cb.grid(row=0, column=3, sticky="ew", padx=8)

        self.usar_mic = tk.BooleanVar(value=True)
        self.usar_pc = tk.BooleanVar(value=True)
        ttk.Label(cfg, text="Traducir a", style="Campo.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.destino_cb = ttk.Combobox(cfg, values=list(DESTINOS), state="readonly", width=28)
        self.destino_cb.current(list(DESTINOS).index("Español"))
        self.destino_cb.grid(row=1, column=1, sticky="ew", padx=(8, 20), pady=(12, 0))
        opciones = tk.Frame(cfg, bg=C_TARJETA)
        opciones.grid(row=1, column=2, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Checkbutton(opciones, text="Mi micrófono (Yo)", variable=self.usar_mic).pack(side="left")
        ttk.Checkbutton(opciones, text="Audio de la PC (Ellos)", variable=self.usar_pc).pack(
            side="left", padx=(16, 0))

        tk.Frame(panel, bg=C_BORDE, height=1).pack(fill="x")
        botones = tk.Frame(panel, bg=C_TARJETA)
        botones.pack(fill="x", padx=16, pady=10)
        self.btn_ini = ttk.Button(botones, text="▶  Iniciar", style="Principal.TButton", command=self.iniciar)
        self.btn_ini.pack(side="left")
        self.btn_fin = ttk.Button(botones, text="■  Detener", style="Peligro.TButton",
                                  command=self.detener, state="disabled")
        self.btn_fin.pack(side="left", padx=8)
        ttk.Button(botones, text="📂  Abrir carpeta", style="Suave.TButton",
                   command=self.abrir_carpeta).pack(side="left")
        self.lbl_cuenta = tk.Label(botones, text="0 mensajes", bg=C_TARJETA, fg=C_SUAVE, font=(FUENTE, 9))
        self.lbl_cuenta.pack(side="right")

        # ---- transcripcion ----
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        marco = tk.Frame(paned, bg=C_TARJETA, highlightthickness=1, highlightbackground=C_BORDE)
        paned.add(marco, weight=3)
        self._panel_ia(paned)
        tk.Label(marco, text="TRANSCRIPCIÓN", bg=C_TARJETA, fg=C_SUAVE,
                 font=(FUENTE + " Semibold", 9)).pack(anchor="w", padx=18, pady=(10, 6))
        tk.Frame(marco, bg=C_BORDE, height=1).pack(fill="x")
        cuerpo = tk.Frame(marco, bg=C_PELO)
        cuerpo.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(cuerpo, orient="vertical")
        sb.pack(side="right", fill="y")
        self.texto = tk.Text(cuerpo, wrap="none", font=(FUENTE, 11), bg=C_PELO, fg=C_TEXTO,
                             relief="flat", borderwidth=0, highlightthickness=0, padx=14, pady=10,
                             yscrollcommand=sb.set, cursor="arrow", state="disabled")
        sb.config(command=self.texto.yview)
        self.texto.pack(side="left", fill="both", expand=True)
        self.texto.bind("<Configure>", self._ajustar)
        self.texto.tag_config("vacio", foreground=C_SUAVE, justify="center", spacing1=110,
                              font=(FUENTE, 12))
        self._mostrar_vacio()

        # ---- barra de estado ----
        barra = tk.Frame(root, bg=C_FONDO)
        barra.pack(fill="x", padx=16, pady=(0, 10))
        self.punto = tk.Canvas(barra, width=12, height=12, bg=C_FONDO, highlightthickness=0)
        self.punto.pack(side="left", padx=(2, 6))
        self._led = self.punto.create_oval(2, 2, 10, 10, fill=ESTADOS["idle"], outline="")
        self.estado = ttk.Label(barra, text="Listo. Usa audífonos para evitar que se duplique lo que dicen.",
                                style="Estado.TLabel")
        self.estado.pack(side="left", fill="x")

        root.protocol("WM_DELETE_WINDOW", self.cerrar)

    # ---------- asistente IA ----------
    def _panel_ia(self, paned):
        marco = tk.Frame(paned, bg=C_TARJETA, highlightthickness=1, highlightbackground=C_BORDE)
        paned.add(marco, weight=2)
        cab = tk.Frame(marco, bg=C_TARJETA)
        cab.pack(fill="x", padx=18, pady=(10, 6))
        tk.Label(cab, text="ASISTENTE IA", bg=C_TARJETA, fg=C_SUAVE, font=(FUENTE + " Semibold", 9)).pack(side="left")
        tk.Label(cab, text="local · gratis", bg=C_TARJETA, fg=C_PASTO, font=(FUENTE, 9)).pack(side="right")
        tk.Frame(marco, bg=C_BORDE, height=1).pack(fill="x")

        rapidos = tk.Frame(marco, bg=C_TARJETA)
        rapidos.pack(fill="x", padx=12, pady=(8, 2))
        for txt, preg in (("Resumen", "Hazme un resumen de la llamada."),
                          ("Puntos clave", "Dame los puntos clave de la llamada en una lista."),
                          ("Pendientes", "¿Qué tareas o acuerdos pendientes hay?")):
            ttk.Button(rapidos, text=txt, style="Suave.TButton",
                       command=lambda q=preg: self.preguntar_ia(q)).pack(side="left", padx=3)

        cuerpo = tk.Frame(marco, bg=C_PELO)
        cuerpo.pack(fill="both", expand=True, pady=(6, 0))
        sb = ttk.Scrollbar(cuerpo, orient="vertical")
        sb.pack(side="right", fill="y")
        self.chat = tk.Text(cuerpo, wrap="word", font=(FUENTE, 10), bg=C_PELO, fg=C_TEXTO, relief="flat",
                            borderwidth=0, highlightthickness=0, padx=14, pady=10, state="disabled",
                            yscrollcommand=sb.set, cursor="arrow", spacing3=4)
        sb.config(command=self.chat.yview)
        self.chat.pack(side="left", fill="both", expand=True)
        self.chat.tag_config("preg", foreground="white", background=C_CIELO, lmargin1=8, lmargin2=8,
                             rmargin=8, spacing1=10, spacing3=6, font=(FUENTE + " Semibold", 10))
        self.chat.tag_config("resp", foreground=C_TEXTO, lmargin1=8, lmargin2=8, rmargin=8)
        self.chat.tag_config("nota", foreground=C_SUAVE, justify="center", spacing1=60, font=(FUENTE, 10))
        self._chat_nota("🐾\nPregúntale a Tole Tole sobre la llamada.\nFunciona en tu PC, sin gastar nada.")

        fila = tk.Frame(marco, bg=C_TARJETA)
        fila.pack(fill="x", padx=12, pady=10)
        self.ia_entrada = ttk.Entry(fila, font=(FUENTE, 10))
        self.ia_entrada.pack(side="left", fill="x", expand=True, ipady=5)
        self.ia_entrada.bind("<Return>", lambda e: self.preguntar_ia())
        self.btn_ia = ttk.Button(fila, text="Enviar", style="Principal.TButton", command=self.preguntar_ia)
        self.btn_ia.pack(side="left", padx=(8, 0))

    def _chat_nota(self, t):
        self.chat.config(state="normal")
        self.chat.insert("end", t, "nota")
        self.chat.config(state="disabled")
        self._chat_vacio = True

    def _chat_add(self, texto, tag):
        def _u():
            self.chat.config(state="normal")
            if getattr(self, "_chat_vacio", False):
                self.chat.delete("1.0", "end")
                self._chat_vacio = False
            self.chat.insert("end", texto, tag)
            self.chat.config(state="disabled")
            self.chat.see("end")
        self.root.after(0, _u)

    def preguntar_ia(self, pregunta=None):
        pregunta = (pregunta or self.ia_entrada.get()).strip()
        if not pregunta or self.ia_ocupada:
            return
        self.ia_entrada.delete(0, "end")
        self.ia_ocupada = True
        self.btn_ia.config(state="disabled")
        self._chat_add(f" {pregunta} \n", "preg")
        threading.Thread(target=self._responder, args=(pregunta,), daemon=True).start()

    def _responder(self, pregunta):
        try:
            if not self.ia_listo:
                self._chat_add("Preparando la IA local...\n", "resp")
                self.ia.preparar(self.set_estado)
                self.ia_listo = True
                self.set_estado("IA lista.")
            tx = "\n".join(f"[{h}] {q}: {t}" for h, q, t in self.historial_tx)
            r = self.ia.preguntar(tx, pregunta, self.ia_chat, lambda x: self._chat_add(x, "resp"))
            self.ia_chat += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": r}]
            self._chat_add("\n", "resp")
            if self.archivo and not self.archivo.closed:
                self.archivo.write(f"\n[IA] Pregunta: {pregunta}\n[IA] Respuesta: {r}\n\n")
                self.archivo.flush()
        except Exception as e:
            self._chat_add(f"No se pudo usar la IA: {e}\n", "resp")
        finally:
            self.ia_ocupada = False
            self.root.after(0, lambda: self.btn_ia.config(state="normal"))

    # ---------- utilidades UI ----------
    def _estilos(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox", padding=6, fieldbackground="white", background="white",
                    bordercolor=C_BORDE, lightcolor=C_BORDE, darkcolor=C_BORDE, arrowcolor=C_CIELO_OSC)
        s.map("TCombobox", fieldbackground=[("readonly", "white"), ("disabled", C_FONDO)],
              foreground=[("disabled", C_SUAVE)])
        s.configure("Campo.TLabel", background=C_TARJETA, foreground=C_SUAVE, font=(FUENTE + " Semibold", 10))
        s.configure("Estado.TLabel", background=C_FONDO, foreground=C_SUAVE, font=(FUENTE, 9))
        s.configure("TCheckbutton", background=C_TARJETA, foreground=C_TEXTO, font=(FUENTE, 10))
        s.map("TCheckbutton", background=[("active", C_TARJETA)])
        base = dict(font=(FUENTE + " Semibold", 10), padding=(18, 8), borderwidth=0, focusthickness=0)
        s.configure("Principal.TButton", background=C_CIELO, foreground="white", **base)
        s.map("Principal.TButton", background=[("active", C_CIELO_OSC), ("disabled", "#b5dcde")],
              foreground=[("disabled", "white")])
        s.configure("Peligro.TButton", background="#d9534f", foreground="white", **base)
        s.map("Peligro.TButton", background=[("active", "#b83f3b"), ("disabled", "#efbcba")],
              foreground=[("disabled", "white")])
        s.configure("Suave.TButton", background="#e8efef", foreground=C_TEXTO, **base)
        s.map("Suave.TButton", background=[("active", "#d6e2e2")])
        s.configure("Vertical.TScrollbar", background="#b9cfd0", troughcolor=C_PELO,
                    bordercolor=C_PELO, arrowcolor=C_PELO, relief="flat")
        s.map("Vertical.TScrollbar", background=[("active", C_CIELO)])

    def _rueda(self, w):
        """Que la rueda del mouse mueva la transcripcion aunque este sobre una tarjeta."""
        w.bind("<MouseWheel>", lambda e: (self.texto.yview_scroll(int(-e.delta / 120) * 3, "units"), "break")[1])
        for h in w.winfo_children():
            self._rueda(h)

    def _ajustar(self, _e=None):
        ancho = max(200, self.texto.winfo_width() - 130)
        for lb in self.etiquetas:
            lb.config(wraplength=ancho)

    def _mostrar_vacio(self):
        self.texto.config(state="normal")
        self.texto.insert("end", "🐾\nAquí aparecerá la conversación.\nPresiona “Iniciar” para comenzar.", "vacio")
        self.texto.config(state="disabled")
        self._vacio = True

    def set_estado(self, t):
        def _u():
            self.estado.config(text=t)
            if t.startswith("Escuchando"):
                modo = "live"
            elif t.startswith("Error"):
                modo = "error"
            elif t.startswith(("Cargando", "Terminando")):
                modo = "busy"
            else:
                modo = "idle"
            self.punto.itemconfig(self._led, fill=ESTADOS[modo])
        self.root.after(0, _u)

    def agregar_linea(self, hora, quien, texto, traduccion=None, etiqueta="ES"):
        def _u():
            t = self.texto
            t.config(state="normal")
            if self._vacio:
                t.delete("1.0", "end")
                self._vacio = False
            color = C_CIELO if quien == "Yo" else C_PASTO
            nombre = "Yo" if quien == "Yo" else "Ellos"
            ancho = max(200, t.winfo_width() - 130)

            # tarjeta: franja de color a la izquierda + contenido
            tarjeta = tk.Frame(t, bg=C_TARJETA, highlightthickness=1, highlightbackground=C_BORDE)
            tk.Frame(tarjeta, bg=color, width=6).pack(side="left", fill="y")
            cont = tk.Frame(tarjeta, bg=C_TARJETA)
            cont.pack(side="left", fill="both", expand=True, padx=14, pady=10)

            cab = tk.Frame(cont, bg=C_TARJETA)
            cab.pack(fill="x")
            tk.Label(cab, text=f"  {nombre}  ", bg=color, fg="white",
                     font=(FUENTE + " Semibold", 9)).pack(side="left")
            tk.Label(cab, text=hora, bg=C_TARJETA, fg="#93a5a7", font=("Consolas", 9)).pack(side="left", padx=10)

            lb = tk.Label(cont, text=texto, bg=C_TARJETA, fg=C_TEXTO, font=(FUENTE, 11),
                          justify="left", anchor="w", wraplength=ancho)
            lb.pack(fill="x", pady=(6, 0))
            self.etiquetas.append(lb)

            if traduccion:
                caja = tk.Frame(cont, bg=C_TRAD_BG)
                caja.pack(fill="x", pady=(8, 0))
                tk.Label(caja, text=etiqueta, bg=C_TRAD, fg="white",
                         font=(FUENTE + " Semibold", 8), padx=5).pack(side="left", anchor="n", padx=(6, 0), pady=7)
                lt = tk.Label(caja, text=traduccion, bg=C_TRAD_BG, fg=C_TRAD, font=(FUENTE, 11, "italic"),
                              justify="left", anchor="w", wraplength=ancho - 40)
                lt.pack(side="left", fill="x", expand=True, padx=8, pady=6)
                self.etiquetas.append(lt)

            self._rueda(tarjeta)
            t.window_create("end", window=tarjeta, stretch=1)
            t.insert("end", "\n")
            t.config(state="disabled")
            t.update_idletasks()
            t.see("end")
            self.n_mensajes += 1
            self.lbl_cuenta.config(text=f"{self.n_mensajes} mensaje" + ("" if self.n_mensajes == 1 else "s"))
        self.root.after(0, _u)

    def error(self, msg):
        self.root.after(0, lambda: messagebox.showerror("Error", msg))

    def abrir_carpeta(self):
        os.makedirs(CARPETA, exist_ok=True)
        os.startfile(CARPETA)

    # ---------- control ----------
    def iniciar(self):
        if not (self.usar_mic.get() or self.usar_pc.get()):
            messagebox.showwarning("Nada que escuchar", "Activa al menos una fuente de audio.")
            return
        self.btn_ini.config(state="disabled")
        for w in (self.idioma, self.modelo_cb, self.destino_cb):
            w.config(state="disabled")
        threading.Thread(target=self._arrancar, daemon=True).start()

    def _arrancar(self):
        nombre = MODELOS[self.modelo_cb.get()]
        if self.modelo is None or self.modelo_nombre != nombre:
            self.set_estado(f"Cargando modelo '{nombre}' (la primera vez se descarga, puede tardar)...")
            try:
                self.modelo, self.dispositivo = cargar_modelo(nombre)
                self.modelo_nombre = nombre
            except Exception as e:
                self.error(f"No se pudo cargar el modelo:\n{e}")
                self.root.after(0, self._reset_botones)
                return

        self.destino = DESTINOS[self.destino_cb.get()]
        self.traductor_activo = self.destino is not None
        if self.traductor_activo and self.traductor is None:
            self.traductor = Traductor()

        os.makedirs(CARPETA, exist_ok=True)
        ruta = os.path.join(CARPETA, datetime.datetime.now().strftime("llamada_%Y-%m-%d_%H-%M-%S.txt"))
        self.archivo = open(ruta, "a", encoding="utf-8")
        self.archivo.write(f"Transcripcion iniciada {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        self.archivo.flush()

        self.cola = queue.Queue()
        self.historial_tx, self.ia_chat = [], []
        self.activo.set()
        sel = IDIOMAS[self.idioma.get()]
        self.lang = None if sel in (None, "auto") else sel
        self.restringir = sel is None   # solo es/en en el modo automatico clasico
        threading.Thread(target=self._transcribir, args=(self.cola,), daemon=True).start()

        try:
            if self.pa is None:
                self.pa = pyaudio.PyAudio()
            mic, loop = dispositivos(self.pa)
            if self.usar_mic.get():
                Capturador("Yo", self.pa, mic, self.cola, self.activo, self.error).start()
            if self.usar_pc.get():
                if loop is None:
                    self.error("No se encontro el audio de salida de la PC (loopback).")
                else:
                    Capturador("Ellos", self.pa, loop, self.cola, self.activo, self.error).start()
        except Exception as e:
            self.error(f"No se pudo abrir el audio:\n{e}")

        self.root.after(0, lambda: self.btn_fin.config(state="normal"))
        self.set_estado(f"Escuchando (usando {self.dispositivo})... guardando en {ruta}")

    def _transcribir(self, cola):
        while True:
            item = cola.get()
            if item is None:
                break
            inicio, quien, audio = item
            try:
                texto, idioma = self._whisper(audio)
            except Exception as e:
                self.set_estado(f"Error al transcribir: {e}")
                continue
            if not texto:
                continue
            traduccion = None
            if self.traductor_activo and idioma != self.destino:
                try:
                    traduccion = self.traductor.traducir(texto, idioma, self.destino, self.set_estado)
                except Exception as e:
                    self.set_estado(f"Error al traducir: {e}")
            hora = inicio.strftime("%H:%M:%S")
            if self.archivo and not self.archivo.closed:
                extra = f"   =>  [{self.destino.upper()}] {traduccion}" if traduccion else ""
                self.archivo.write(f"[{hora}] {quien}: {texto}{extra}\n")
                self.archivo.flush()
            self.historial_tx.append((hora, quien, texto))
            self.agregar_linea(hora, quien, texto, traduccion, (self.destino or "").upper())
            pend = cola.qsize()
            if self.activo.is_set():
                self.set_estado(f"Escuchando ({self.dispositivo})..."
                                + (f"  ({pend} fragmentos en espera)" if pend else ""))

    def _whisper(self, audio):
        # En GPU sobra velocidad para la busqueda mas precisa; en CPU se usa la rapida
        beam = 5 if self.dispositivo == "GPU" else 1
        texto, idioma = self._pasada(audio, beam, vad=True)
        if not texto:
            # El filtro de voz suele descartar canto o voz con musica de fondo:
            # reintentamos sin filtro y solo aceptamos lo que Whisper reconoce con confianza.
            texto, idioma = self._pasada(audio, beam, vad=False)
        return texto, idioma

    def _pasada(self, audio, beam, vad):
        opts = dict(beam_size=beam, vad_filter=vad, condition_on_previous_text=False)
        segs, info = self.modelo.transcribe(audio, language=self.lang, **opts)
        segs = list(segs)
        idioma = self.lang or info.language
        if self.lang is None and self.restringir and info.language not in ("es", "en"):
            # Solo permitimos espanol o ingles: elegimos el mas probable de los dos
            probs = dict(info.all_language_probs or [])
            idioma = "es" if probs.get("es", 0) >= probs.get("en", 0) else "en"
            segs = list(self.modelo.transcribe(audio, language=idioma, **opts)[0])
        if not vad:
            segs = [s for s in segs if s.avg_logprob > -1.0 and s.no_speech_prob < 0.8]
        partes = [s.text.strip() for s in segs if not es_alucinacion(s.text, estricto=not vad)]
        return " ".join(p for p in partes if p).strip(), idioma

    def detener(self):
        self.activo.clear()
        self.btn_fin.config(state="disabled")
        self.set_estado("Terminando de transcribir lo pendiente...")
        cola = self.cola

        def _fin():
            import time
            time.sleep(BLOQUE * 2)   # dejar que los capturadores envien lo ultimo
            cola.put(None)
            while not cola.empty():
                time.sleep(0.2)
            time.sleep(1)
            if self.archivo:
                self.archivo.close()
            self.set_estado(f"Detenido. Transcripcion guardada en {CARPETA}")
            self.root.after(0, self._reset_botones)
        threading.Thread(target=_fin, daemon=True).start()

    def _reset_botones(self):
        self.btn_ini.config(state="normal")
        self.btn_fin.config(state="disabled")
        for w in (self.idioma, self.modelo_cb, self.destino_cb):
            w.config(state="readonly")

    def cerrar(self):
        self.ia.cerrar()
        self.activo.clear()
        if self.archivo and not self.archivo.closed:
            self.archivo.close()
        self.root.destroy()


if __name__ == "__main__":
    try:  # que Windows muestre el icono propio en la barra de tareas
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ToleTole.TranscriptorLlamadas")
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()
