# 🐾 Tole Tole · Transcriptor de Llamadas

Transcribe en vivo tu micrófono ("Yo") y el audio de la PC ("Ellos") con Whisper, todo en local.
Guarda la conversación en `Documentos/Transcripciones`.

## Funciones
- **Transcripción en vivo** en más de 20 idiomas (español, inglés, francés, alemán, japonés...), con detección automática.
- **Traducción sin internet** a otro idioma, con modelos de Argos Translate. Cada par se descarga la primera vez y,
  si no hay modelo directo, traduce pasando por inglés.
- **Asistente IA local** (Ollama + `qwen2.5:3b`): pregunta sobre la llamada, pide un resumen o los pendientes.
  Es gratis, no usa internet ni gasta tokens. Ollama se abre solo cuando haces una pregunta y se cierra al salir.
- Tarjetas por mensaje con la hora, quién habló y la traducción.

## Instalación
```
pip install -r requirements.txt
python transcriptor.py
```
O ejecuta `iniciar.bat`. Con `crear_acceso.ps1` creas un acceso directo en el Escritorio con el icono de Tole Tole.

Para el asistente IA instala [Ollama](https://ollama.com) (`winget install Ollama.Ollama`). El modelo (~2 GB)
se descarga solo la primera vez que hagas una pregunta.

## Notas
- Usa audífonos para evitar que se duplique el audio.
- Graba conversaciones solo con el consentimiento de los participantes, según la ley de tu país.
- Las respuestas de la IA pueden equivocarse, sobre todo en llamadas largas: verifica los datos importantes.
