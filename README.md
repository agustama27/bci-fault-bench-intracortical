# bci-fault-bench-intracortical

Prueba de concepto (PoC 1) de una **línea futura** del TFG sobre inyección de fallas en pipelines BCI basados en Lab Streaming Layer (LSL): llevar el banco de pruebas, hoy pensado para EEG, a **señales intracorticales** (conteos de spikes).

Esta PoC es **offline**, sin LSL todavía. Establece dos cosas que la etapa siguiente necesita:

1. Un **loader** que convierte una sesión pública de O'Doherty et al. (2017) en conteos binneados por electrodo y velocidad del cursor.
2. Dos **decodificadores de referencia**, Wiener y Kalman, escritos desde cero en numpy, con una métrica de base contra la cual medir después el efecto de las fallas.

> Estado: exploratorio. Una sola sesión, un solo sujeto, sin inferencia confirmatoria.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `src/icbench/data.py` | Lee el `.mat` v7.3 con `h5py`, binnea spikes (20 ms por defecto), suma las unidades de cada electrodo (96 canales), calcula velocidad y posición del cursor en la grilla de bins y guarda un `.npz` |
| `src/icbench/decoders.py` | `WienerDecoder` (ridge sobre la historia causal de K bins) y `KalmanDecoder` (Wu et al., 2006) con manejo de bins faltantes |
| `scripts/poc1_offline.py` | Validación cruzada temporal de 5 folds contiguos, R² y r de Pearson por eje, chequeo de bins faltantes, CSV y figura |

### Decisiones que conviene conocer

- **Unidad hash incluida por defecto.** En el dataset, la fila 0 de `spikes` es la unidad sin clasificar (*hash*). Se suma junto con las unidades clasificadas porque así se parece más a lo que consume un BCI en tiempo real (cruces de umbral por electrodo, sin *spike sorting*). Con `--no-include-hash` se queda solo con las unidades clasificadas.
- **Velocidad = diferencia finita de la posición interpolada en los bordes de cada bin**, o sea, la velocidad media del bin, en mm/s.
- **Spikes fuera de `[t[0], t[-1]]` se descartan.** Hay spikes que arrancan unos 2,5 s antes que la cinemática.
- **Celdas vacías.** MATLAB las guarda como `uint64 [0, 0]` con el atributo `MATLAB_empty`; el loader las trata como unidades sin spikes.
- **Wiener**: historia causal de K = 10 bins (200 ms), ridge con α = 1000 sobre conteos estandarizados y suavizado gaussiano causal (σ = 2 bins).
- **Kalman**: estado `[pos, vel, acc, 1]`. La constante absorbe el intercepto de H. A, W, H y Q salen por mínimos cuadrados, las observaciones son √conteos y el desfase neural es de 3 bins (60 ms).
- **Bins faltantes** (filas NaN). El Kalman hace **solo el paso de predicción** (`x = A x`, `P = A P Aᵀ + W`). El Wiener los ve como **cero spikes**, que es el comportamiento ingenuo de un decodificador sin estado. Esta asimetría es el gancho para la línea de inyección de fallas.

## Cómo correrlo

Requiere Python 3.12. Probado en Windows 11 con Git Bash.

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock.txt # versiones exactas; requirements.txt tiene las mínimas
pip install -e .

# 1) Descargar la sesión (80 MiB) a data/raw/
curl -L -o data/raw/indy_20161005_06.mat "https://zenodo.org/records/583331/files/indy_20161005_06.mat?download=1"
#    MD5 esperado: 5ea300952642e0fc54245144499db9bb

# 2) Binnear -> data/processed/indy_20161005_06_bin20ms.npz
python -m icbench.data --bin-ms 20              # --no-include-hash para excluir la unidad 0

# 3) PoC offline -> results/poc1_metrics.csv y results/poc1_trace.png
python scripts/poc1_offline.py
```

`data/raw/`, `data/processed/` y `results/` no se versionan. El script tarda unos **12 s** en CPU (el loader, ~2 s).

## Resultados

Sesión `indy_20161005_06`: 374 s, 18.700 bins de 20 ms, 96 electrodos (M1), 249 unidades (con hash) y 312.419 spikes dentro del rango. La tasa media es de 8,7 Hz por canal y hay 11 canales silenciosos.

### Validación cruzada temporal (5 folds contiguos, sin mezclar)

Media sobre los 5 folds, sin bins faltantes:

| Decodificador | R² x | R² y | r x | r y |
|---|---|---|---|---|
| Wiener | 0,449 | 0,585 | 0,680 | 0,770 |
| Kalman | 0,378 | 0,425 | 0,636 | 0,672 |

Por fold, el R² medio de los dos ejes va de 0,47 a 0,56 para Wiener y de 0,35 a 0,43 para Kalman. El detalle está en `results/poc1_metrics.csv`.

### Chequeo de bins faltantes

Se ponen en NaN un 5 % y un 20 % de los bins de test, elegidos al azar (semilla fija). La tabla muestra el cambio del R² medio (x, y) contra la condición limpia:

| Decodificador | 5 % faltante | 20 % faltante |
|---|---|---|
| Wiener (relleno con ceros) | −0,007 | −0,042 |
| Kalman (solo predicción) | −0,006 | −0,027 |

Lectura: el Wiener **rinde más en limpio**, pero el Kalman **se degrada menos** cuando faltan bins, porque su estado absorbe los huecos. Esta es justamente la confusión entre "tipo de señal" y "decodificador con estado" que hay que declarar en la línea de fallas.

![Velocidad real vs decodificada, extracto de 20 s](results/poc1_trace.png)

*(La figura se genera localmente en `results/`, que no se versiona.)*

### Caveats

- **Hiperparámetros elegidos a mano** en un único split exploratorio (test = 60–80 % de la sesión, que coincide con el fold 4). Hay un sesgo leve de selección; no hay CV anidada.
- En los folds intermedios, el set de entrenamiento no es contiguo. La única transición espuria (el salto entre los dos tramos) entra en la estimación de A y de los rezagos del Wiener. Con unos 15.000 bins de entrenamiento, el efecto es despreciable.
- Los bins faltantes son **uniformes e independientes**, no ráfagas ni cortes. Es un chequeo de cordura, no el modelo de fallas del banco.
- R² bajo el nivel publicado para decodificadores modernos. No es el objetivo: acá se busca una línea base reproducible y simple.

## Qué sigue

1. **Replay LSL de los conteos binneados**: un *outlet* de 96 canales a 50 Hz (`nominal_srate = 50`, tipo `float32` o `int16`) que reproduzca la sesión en tiempo real, y un *inlet* que decodifique en línea con los mismos modelos.
2. **Inyección de fallas en tiempo físico (ms)**: pérdida, retraso, jitter y cortes aplicados sobre el stream. Las severidades se expresan en ms y en bins equivalentes, para poder comparar con EEG usando la misma traza de enlace.
3. Curvas dosis-respuesta de R² y r contra la severidad, con Wiener (sin estado) y Kalman (con estado) lado a lado.

## Referencias

O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., y Sabes, P. N. (2017). *Nonhuman primate reaching with multichannel sensorimotor cortex electrophysiology* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.583331 (licencia CC BY 4.0)

Wu, W., Gao, Y., Bienenstock, E., Donoghue, J. P., y Black, M. J. (2006). Bayesian population decoding of motor cortical activity using a Kalman filter. *Neural Computation, 18*(1), 80–118. https://doi.org/10.1162/089976606774841585

Ambos DOI se verificaron el 2026-09-27: el de Wu et al. contra la API de Crossref y el de Zenodo contra la de DataCite.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos pertenecen a sus autores (CC BY 4.0) y no se redistribuyen en este repositorio.
