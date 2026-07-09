# Handoff — Segundo dataset (clasificador de flush stage / WHEN)

> Contexto para retomar el proyecto en otra chat y avanzar **el segundo dataset**.
> Documento local (en `_private/`, no se sube). Fecha: 2026-07-09.

## Orientación rápida del proyecto

- **Repo:** `C:\Users\AngelTatianaMendozaB\OneDrive - ITS Angelo Rizzoli\Desktop\G\TINAT` → GitHub `angelbohorquez05/TINAT`.
- **Contexto completo:** leer `CLAUDE.md` del repo al empezar (reglas de anonimización, idiomas, workflow git).
- **Setup dos capas:** repo TINAT anonimizado (lo que se sube / examen) + `_private/` local (real, nunca se sube). No sacar nombres reales/empresa fuera de `_private/`.
- **Idiomas:** docs/README en italiano, código/commits en inglés, conversación con Angel en español.
- **Ubicación:** el proyecto vive ahora en `Desktop\G\TINAT` (dentro de OneDrive). ⚠️ OneDrive
  sincronizando `.git` puede corromperlo — red de seguridad: todo lo commiteado está en GitHub
  (re-clonar si pasa). Recomendable pausar OneDrive durante operaciones git largas.
- **Material real fusionado (sin duplicar código):** datos reales en `data/` y pesos/entrenamiento
  en `models/` (raíz, gitignored); docs con nombres reales en `_private/empresa/docs_real/`
  (`HELOPELTIS_DETECTOR_CARD.md`, `NEXT_STEPS_WORKSTREAM_B.md`) y material viejo en
  `_private/empresa/_archived/`. El código vive una sola vez en `src/`.

## Estado actual (qué está hecho / qué falta)

Hecho:
- Detector Helopeltis (YOLOv8n) — notebook `01`, entrenado sobre dataset público.
- Métrica WHO×WHERE×WHEN + capa de fusión + demo end-to-end (detector simulado). Tests pytest verdes + CI.
- DevOps: CI (GitHub Actions), Docker, GitHub Flow + PR template, model card, decision-log.

Falta (esta tarea):
- **Segundo dataset → entrenar el clasificador de flush stage (WHEN).** Sin él, `FusionEngine` usa un stage fijo (fallback). Es lo que desbloquea la dimensión WHEN de verdad.

## La tarea concreta: entrenar el flush-stage classifier

**Objetivo:** producir `models/flush_stage.tflite` para que `FlushClassifier` (en `src/fusion.py`) lo cargue solo.

**Clases (orden EXACTO, no cambiar):** `["dormant", "new_flush", "mature_leaf"]`
- `dormant` (0–2 meses, riesgo bajo)
- `new_flush` (3–4 meses, **riesgo ALTO** — ventana Helopeltis) → la clase que más importa acertar
- `mature_leaf` (5+ meses, riesgo bajo)

**Modelo:** MobileNetV2 (ImageNet) + cabeza de 3 clases, 224×224, normalización ImageNet
(mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`). Todo ya definido en el notebook `02`.

**Dónde va el dataset:**
```
data/raw/flush_stages/
    dormant/
    new_flush/
    mature_leaf/
```
Meta: ≥ 150 imágenes por clase. (`data/` está en `.gitignore` — no se versiona.)

**Fuente del dataset (ya identificado):**
- **Mendeley — TeaLeafAgeQuality:** https://data.mendeley.com/datasets/7t964jmmy3/1
- 2.208 imágenes de *Camellia sinensis*, 4 clases de edad **T1–T4**, licencia **CC BY 4.0**.
- **Mapeo a nuestras 3 clases:** `T1+T2 → new_flush/`, `T3+T4 → mature_leaf/`.
  `dormant` NO está en Mendeley → añadir a mano desde **iNaturalist** (*Camellia sinensis* sin
  brotes nuevos). Alternativa: si no se consigue `dormant`, entrenar 2 clases (new_flush vs
  mature_leaf) y añadir dormant después.
- **Estado:** dataset identificado pero **NO descargado**. Las carpetas destino ya existen vacías
  (0 imágenes) en `data/raw/flush_stages/{dormant,new_flush,mature_leaf}/` (raíz del repo,
  gitignored). Solo falta bajar las imágenes de Mendeley y ordenarlas ahí con el mapeo de arriba.

## Pasos (el notebook 02 ya está andamiado — solo descomentar por secciones)

1. Bajar el dataset y ordenarlo en `data/raw/flush_stages/{clase}/`.
2. `02` §2 *Data check* → confirmar ≥150/clase.
3. `02` §3 *Split* (descomentar, correr una vez) → crea `train/val/test`.
4. `02` §5 DataLoaders → §7 Training (frozen backbone → fine-tuning) → §8 Eval.
   - Objetivo: **val accuracy ≥ 85% en `new_flush`** (fallar la clase de alto riesgo es lo caro).
5. `02` §9 export → obtener `models/flush_stage.tflite`.
6. Verificar integración: correr `python -m src.demo` — `FlushClassifier.available` debe dar `True`
   y el stage salir del modelo, no del fallback.

## Gotchas / cosas a revisar (no dar por sentado)

- **Export a TFLite:** el notebook `02` sugiere `tf2onnx`, que va TF→ONNX (¡al revés!). La ruta
  correcta es exportar a ONNX y convertir con **`onnx2tf`**:
  `pip install onnx2tf && onnx2tf -i models/flush_stage.onnx -o models/flush_stage.tflite`.
  `FlushClassifier.predict` espera un TFLite con entrada **NHWC** (1,224,224,3) y normalización
  ImageNet; confirmar que el `.tflite` final coincide (formato de entrada y orden de clases
  `dormant,new_flush,mature_leaf`) o ajustar el wrapper. Corregir de paso esa línea del notebook.
- **models/** puede no existir aún y está fuera de git (pesos no versionados) — crearlo local.
- **Decision-log:** registrar la elección del dataset (fuente, por qué ese, licencia) en
  `docs/decision-log.md` — es la regla de oro del curso.
- **Antes de commitear:** `grep` de nombres reales (tanit|jerome|nenad|anyanwu|green global). El
  código del clasificador es público/genérico, así que va al repo normal; solo este handoff se
  queda en `_private/`.

## Prompt sugerido para la nueva chat

> Estoy retomando el proyecto TINAT (repo en `PROGGETTO\DevOps\TINAT`). Lee el `CLAUDE.md` y
> `_private/notas/handoff-segundo-dataset.md`. Quiero avanzar el **segundo dataset**: entrenar el
> clasificador de flush stage (WHEN) y generar `models/flush_stage.tflite`. El dataset es
> Mendeley TeaLeafAgeQuality (https://data.mendeley.com/datasets/7t964jmmy3/1): mapear
> T1+T2→new_flush, T3+T4→mature_leaf, y dormant desde iNaturalist. Empecemos por bajarlo y
> ordenarlo en `data/raw/flush_stages/`.
