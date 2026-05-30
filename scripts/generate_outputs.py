"""
Simulación Montecarlo — Sistema de embarcaciones turísticas del Lago Titicaca.

Correcciones aplicadas:
- La jornada cierra a t=duracion: no se permiten salidas de embarcaciones después.
- Turistas no atendidos = grupos que quedaron en cola al cierre de la jornada.
- Espera = salida - t_llegada (desde llegada del grupo hasta embarque).
- Utilización sobre ventana fija duracion * n_embarcaciones.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ["notebooks", "figures", "data", "results", "docs", "scripts"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)


# ---------------------------------------------------------------------------
# Ficha de observación sintética
# ---------------------------------------------------------------------------

def save_ficha():
    data = {
        "N_grupo": [1, 2, 3, 4, 5, 6, 7, 8],
        "Hora_llegada": ["09:03", "09:10", "09:18", "09:25", "09:34", "09:40", "09:48", "09:56"],
        "Interllegada_min": [3, 7, 8, 7, 9, 6, 8, 8],
        "Tamano_grupo": [6, 12, 8, 15, 4, 18, 10, 7],
        "Registro_min": [4.1, 5.3, 3.8, 6.2, 3.4, 6.8, 4.7, 5.0],
        "Destino": ["Uros"] * 8,
        "Observacion": [
            "Grupo familiar pequeño",
            "Grupo con guía local",
            "Registro rápido",
            "Pago dividido",
            "Grupo pequeño",
            "Grupo grande",
            "Grupo mixto",
            "Grupo familiar",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(BASE_DIR, "data", "ficha_observacion_sintetica.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Histogramas de validación
# ---------------------------------------------------------------------------

def plot_hist(values, filename, title, xlabel, discrete=False):
    plt.figure(figsize=(8, 4.5))
    sns.histplot(values, kde=not discrete, bins=30, discrete=discrete)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", filename), dpi=160)
    plt.close()


def validate_distributions(rng):
    n = 10_000
    samples = {
        "Interllegadas": rng.exponential(scale=7.5, size=n),
        "Tamano Grupo": np.maximum(1, np.round(rng.triangular(4, 10, 18, size=n)).astype(int)),
        "Registro": rng.triangular(3, 5, 7, size=n),
        "Navegacion Ida": rng.uniform(25, 40, size=n),
        "Permanencia": rng.triangular(45, 90, 180, size=n),
    }
    teoricas = {
        "Interllegadas": 7.5,
        "Tamano Grupo": (4 + 10 + 18) / 3,
        "Registro": (3 + 5 + 7) / 3,
        "Navegacion Ida": (25 + 40) / 2,
        "Permanencia": (45 + 90 + 180) / 3,
    }
    plot_hist(samples["Interllegadas"], "hist_interllegadas.png", "Interllegadas — Exponencial(7.5)", "Minutos")
    plot_hist(samples["Tamano Grupo"], "hist_tamano_grupo.png", "Tamaño de grupo — Triangular discreta(4,10,18)", "Turistas", True)
    plot_hist(samples["Registro"], "hist_registro.png", "Registro — Triangular(3,5,7)", "Minutos")
    plot_hist(samples["Navegacion Ida"], "hist_navegacion.png", "Navegación — Uniforme(25,40)", "Minutos")
    plot_hist(samples["Permanencia"], "hist_permanencia.png", "Permanencia — Triangular(45,90,180)", "Minutos")
    df = pd.DataFrame(
        {
            "Variable": list(samples.keys()),
            "Media_Muestral": [round(np.mean(v), 4) for v in samples.values()],
            "Media_Teorica": [round(teoricas[k], 4) for k in samples.keys()],
            "Varianza_Muestral": [round(np.var(v), 4) for v in samples.values()],
        }
    )
    df.to_csv(os.path.join(BASE_DIR, "results", "validacion_distribuciones.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Generación de grupos
# ---------------------------------------------------------------------------

def generar_grupos(rng, duracion, lambda_grupos):
    """Genera grupos turísticos que llegan durante la jornada (hasta t=duracion)."""
    t = 0.0
    grupos = []
    while True:
        t += rng.exponential(scale=60.0 / lambda_grupos)
        if t > duracion:
            break
        grupos.append(
            {
                "t_llegada": t,
                "tamano": max(1, int(round(rng.triangular(4, 10, 18)))),
                "t_registro": rng.triangular(3, 5, 7),
            }
        )
    # Ventanilla única FIFO: fin de registro acumulado
    fin_ventanilla = 0.0
    for g in grupos:
        inicio = max(g["t_llegada"], fin_ventanilla)
        g["fin_reg"] = inicio + g["t_registro"]
        fin_ventanilla = g["fin_reg"]
    return grupos


# ---------------------------------------------------------------------------
# Simulación de una jornada
# ---------------------------------------------------------------------------

def simular_jornada(
    rng,
    duracion=240,
    lambda_grupos=8,
    n_embarcaciones=3,
    capacidad=30,
    t_prep=10,
    umbral=24,
    espera_max=15,
):
    """
    Simula una jornada completa.

    Correcciones:
    - Embarcaciones sólo parten si salida <= duracion (jornada cerrada a las 10:00).
    - Turistas no atendidos = suma de turistas en cola al cierre.
    - Espera = salida - t_llegada (desde llegada del grupo hasta que la embarcación parte).
    - Utilización sobre duracion * n_embarcaciones (ventana fija de la jornada).
    """
    cola = generar_grupos(rng, duracion, lambda_grupos)
    cola.sort(key=lambda g: g["fin_reg"])
    botes = [0.0] * n_embarcaciones
    esperas = []
    sistemas = []
    viajes = 0
    atendidos = 0
    ocupado = 0.0

    esperas_cola = []  # espera pura en cola de embarque (salida - fin_reg)

    while cola:
        bote_idx = int(np.argmin(botes))
        t_inicio = max(botes[bote_idx], cola[0]["fin_reg"])

        # Cierre de jornada: no se inician más cargas después de t=duracion
        if t_inicio > duracion:
            break

        t = t_inicio
        carga = 0
        cargados = []
        i = 0
        limite = cola[0]["fin_reg"] + espera_max

        while i < len(cola):
            g = cola[i]
            if g["fin_reg"] > max(t, limite):
                break
            t = max(t, g["fin_reg"])
            if carga + g["tamano"] <= capacidad:
                carga += g["tamano"]
                cargados.append(g)
                i += 1
                if carga >= umbral:
                    break
            else:
                break

        if not cargados:
            # Avanzar tiempo del bote al próximo grupo disponible
            botes[bote_idx] = cola[0]["fin_reg"]
            continue

        salida = t if carga >= umbral else max(t, limite)

        # Embarcación sólo parte si alcanza antes del cierre de jornada
        if salida > duracion:
            break

        ciclo_sin_prep = rng.uniform(25, 40) + rng.triangular(45, 90, 180) + rng.uniform(25, 40)
        ciclo_total = ciclo_sin_prep + t_prep

        # Tiempo activo dentro de la ventana de jornada (el barco puede retornar después)
        t_activo_en_ventana = min(salida + ciclo_total, duracion) - salida
        ocupado += t_activo_en_ventana

        botes[bote_idx] = salida + ciclo_total
        viajes += 1
        atendidos += carga

        for g in cargados:
            # Espera desde llegada del grupo hasta embarque (proyecto.md §19)
            espera = salida - g["t_llegada"]
            esperas.append(espera)
            # Espera pura en cola de embarque (desde fin de registro)
            esperas_cola.append(max(0.0, salida - g["fin_reg"]))
            # Tiempo en sistema = espera previa + ciclo de viaje (sin preparación)
            sistemas.append(espera + ciclo_sin_prep)

        cola = cola[len(cargados):]

    # Turistas no atendidos = los que quedaron en cola al cierre de jornada
    no_atendidos = sum(g["tamano"] for g in cola)

    tiempo_total = duracion * n_embarcaciones  # ventana fija
    espera_prom = float(np.mean(esperas)) if esperas else 0.0
    return {
        "Espera_Promedio": espera_prom,
        "Sistema_Promedio": float(np.mean(sistemas)) if sistemas else 0.0,
        "Longitud_Cola": (lambda_grupos / 60.0) * espera_prom,
        "Utilizacion": ocupado / tiempo_total if tiempo_total else 0.0,
        "Porcentaje_Espera": float(np.mean(np.array(esperas_cola) > 0)) if esperas_cola else 0.0,
        "Percentil_95_Espera": float(np.percentile(esperas, 95)) if esperas else 0.0,
        "Turistas_Atendidos": atendidos,
        "Turistas_No_Atendidos": no_atendidos,
        "Viajes": viajes,
        "Ocupacion_Promedio": atendidos / (viajes * capacidad) if viajes else 0.0,
    }


# ---------------------------------------------------------------------------
# Escenarios Montecarlo
# ---------------------------------------------------------------------------

def run_scenarios(rng, n_replicas=1000):
    escenarios = [
        {"Escenario": "E0 Base",               "lambda": 8,  "n_emb": 3, "cap": 30, "prep": 10, "umbral": 24},
        {"Escenario": "E1 Más embarcaciones",   "lambda": 8,  "n_emb": 4, "cap": 30, "prep": 10, "umbral": 24},
        {"Escenario": "E2 Menor preparación",   "lambda": 8,  "n_emb": 3, "cap": 30, "prep":  5, "umbral": 24},
        {"Escenario": "E3 Mayor capacidad",     "lambda": 8,  "n_emb": 3, "cap": 40, "prep": 10, "umbral": 32},
        {"Escenario": "E4 Alta demanda",        "lambda": 10, "n_emb": 3, "cap": 30, "prep": 10, "umbral": 24},
    ]
    rows = []
    for esc in escenarios:
        reps = [
            simular_jornada(
                rng,
                lambda_grupos=esc["lambda"],
                n_embarcaciones=esc["n_emb"],
                capacidad=esc["cap"],
                t_prep=esc["prep"],
                umbral=esc["umbral"],
            )
            for _ in range(n_replicas)
        ]
        df = pd.DataFrame(reps)
        row = {"Escenario": esc["Escenario"]}
        row.update(df.mean(numeric_only=True).round(3).to_dict())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(BASE_DIR, "results", "resultados_escenarios.csv"), index=False)
    return result


# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------

def plot_scenarios(df):
    charts = [
        ("Espera_Promedio",        "escenarios_espera_promedio.png",    "Tiempo promedio de espera por escenario",      "Minutos"),
        ("Percentil_95_Espera",    "escenarios_percentil95.png",        "Percentil 95 de espera por escenario",         "Minutos"),
        ("Utilizacion",            "escenarios_utilizacion.png",        "Utilización promedio de embarcaciones",        "Proporción"),
        ("Turistas_No_Atendidos",  "escenarios_turistas_no_atendidos.png", "Turistas no atendidos por escenario",       "Turistas"),
    ]
    for col, filename, title, ylabel in charts:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="Escenario", y=col, hue="Escenario", legend=False, palette="viridis")
        plt.title(title, fontsize=13)
        plt.ylabel(ylabel)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "figures", filename), dpi=160)
        plt.close()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def write_readme():
    content = """# ModelTitikaka

Simulación Montecarlo del sistema de embarcaciones turísticas del Lago Titicaca, Puno.
Asignatura: SIS230 — Modelado Sistémico y Simulación, UNAP.

## Estructura

```
notebooks/   notebook principal de simulación
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
"""
    with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generando ficha de observación sintética...")
    save_ficha()

    rng = np.random.default_rng(2026)

    print("Validando distribuciones (10 000 muestras)...")
    df_val = validate_distributions(rng)
    print(df_val.to_string(index=False))

    print("\nEjecutando simulación Montecarlo (1 000 réplicas × 5 escenarios)...")
    df_res = run_scenarios(rng, n_replicas=1000)
    print(df_res.to_string(index=False))

    print("\nGenerando gráficos comparativos...")
    plot_scenarios(df_res)

    print("\nActualizando README...")
    write_readme()

    print("\nListo. Archivos generados en:")
    print(f"  {os.path.join(BASE_DIR, 'results')}")
    print(f"  {os.path.join(BASE_DIR, 'figures')}")


if __name__ == "__main__":
    main()
