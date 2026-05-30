# ModelTitikaka

Simulación Montecarlo del sistema de embarcaciones turísticas del Lago Titicaca, Puno.
Asignatura: SIS230 — Modelado Sistémico y Simulación, UNAP.

## Estructura

```
notebooks/   notebook de presentación y análisis de resultados generados
data/        ficha de observación sintética (CSV)
results/     métricas de validación y escenarios (CSV)
figures/     histogramas y gráficos comparativos (PNG)
docs/        informe académico en LaTeX
scripts/     script de generación de resultados
```

## Ejecución rápida

```bash
# Regenerar todos los CSV y figuras
python scripts/generate_outputs.py

# Abrir notebook
jupyter notebook notebooks/simulacion_embarcaciones_titicaca.ipynb
```

## Parámetros del modelo

| Parámetro                | Valor base |
|--------------------------|----------:|
| Duración jornada          | 240 min   |
| Repeticiones Montecarlo   | 1 000     |
| Tasa de llegada           | 8 grupos/h|
| Embarcaciones             | 3         |
| Capacidad por embarcación | 30        |
| Umbral de salida          | 24        |
| Espera máxima salida      | 15 min    |
| Preparación entre viajes  | 10 min    |
| Semilla aleatoria         | 2026      |

## Nota metodológica

La ficha de observación es **sintética y académica**. No corresponde a una medición
presencial oficial; se construyó con base en el PDF del trabajo, fuentes públicas
(MINCETUR, SERNANP) y supuestos operativos coherentes con el sistema real.
