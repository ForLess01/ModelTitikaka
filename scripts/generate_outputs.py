"""
Simulación Montecarlo — Sistema de embarcaciones turísticas del Lago Titicaca.

Correcciones aplicadas:
- La jornada cierra a t=duracion: no se permiten salidas de embarcaciones después.
- Turistas no atendidos = grupos que quedaron en cola al cierre de la jornada.
- Espera = salida - t_llegada (desde llega da del grupo hasta embarque).
- Utilización sobre ventana fija duracion * n_embarcaciones.
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ["notebooks", "figures", "data", "results", "docs", "scripts"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)


# ---------------------------------------------------------------------------
# Generadores por transformada
# ---------------------------------------------------------------------------


def uniform_inverse(rng, low, high, size=None):
    """Uniforme(a,b) mediante transformada inversa: X = a + (b-a)U."""
    u = rng.random(size)
    return low + (high - low) * u


def exponential_inverse(rng, mean, size=None):
    """Exponencial(media) mediante transformada inversa: X = -media ln(1-U)."""
    u = rng.random(size)
    return -mean * np.log1p(-u)


def triangular_inverse(rng, left, mode, right, size=None):
    """Triangular(a,c,b) por transformada inversa de su CDF por tramos."""
    u = rng.random(size)
    split = (mode - left) / (right - left)
    lower = left + np.sqrt(u * (right - left) * (mode - left))
    upper = right - np.sqrt((1.0 - u) * (right - left) * (right - mode))
    value = np.where(u < split, lower, upper)
    return float(value) if size is None else value


def normal_box_muller(rng, mean=0.0, sd=1.0, size=None):
    """Normal mediante Box-Muller; no es inversa de CDF, sino transformada exacta."""
    if size is None:
        u1 = rng.random()
        u2 = rng.random()
        return mean + sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    u1 = rng.random(size)
    u2 = rng.random(size)
    return mean + sd * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def truncated_normal_box_muller(rng, mean, sd, low, high, size):
    """Normal truncada por rechazo, generada desde normales Box-Muller."""
    values = []
    remaining = size
    while remaining > 0:
        draw = normal_box_muller(
            rng, mean, sd, size=max(remaining * 2, 1000) if remaining > 0 else 1000
        )
        draw = np.asarray(draw)
        accepted = draw[(draw >= low) & (draw <= high)]
        values.append(accepted[:remaining])
        remaining -= min(remaining, len(accepted))
    return np.concatenate(values)


def triangular_mean(left, mode, right):
    return (left + mode + right) / 3.0


def triangular_variance(left, mode, right):
    return (
        left**2 + mode**2 + right**2 - left * mode - left * right - mode * right
    ) / 18.0


def triangular_cdf(x, left, mode, right):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    lower = (x > left) & (x <= mode)
    upper = (x > mode) & (x < right)
    y[x >= right] = 1.0
    y[lower] = ((x[lower] - left) ** 2) / ((right - left) * (mode - left))
    y[upper] = 1.0 - ((right - x[upper]) ** 2) / ((right - left) * (right - mode))
    return y


def triangular_pdf(x, left, mode, right):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    lower = (x >= left) & (x <= mode)
    upper = (x > mode) & (x <= right)
    y[lower] = 2.0 * (x[lower] - left) / ((right - left) * (mode - left))
    y[upper] = 2.0 * (right - x[upper]) / ((right - left) * (right - mode))
    return y


def rounded_triangular_moments(left, mode, right):
    values = np.arange(round(left), round(right) + 1)
    lows = np.maximum(left, values - 0.5)
    highs = np.minimum(right, values + 0.5)
    probs = triangular_cdf(highs, left, mode, right) - triangular_cdf(
        lows, left, mode, right
    )
    mean = float(np.sum(values * probs))
    variance = float(np.sum(((values - mean) ** 2) * probs))
    return mean, variance, values, probs


def normal_pdf(x, mean, sd):
    z = (x - mean) / sd
    return np.exp(-0.5 * z**2) / (sd * math.sqrt(2.0 * math.pi))


def normal_cdf_scalar(x, mean, sd):
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def truncated_normal_pdf(x, mean, sd, low, high):
    z = normal_cdf_scalar(high, mean, sd) - normal_cdf_scalar(low, mean, sd)
    y = normal_pdf(x, mean, sd) / z
    return np.where((x >= low) & (x <= high), y, 0.0)


def truncated_normal_moments(mean, sd, low, high):
    alpha = (low - mean) / sd
    beta = (high - mean) / sd
    phi_alpha = math.exp(-0.5 * alpha**2) / math.sqrt(2.0 * math.pi)
    phi_beta = math.exp(-0.5 * beta**2) / math.sqrt(2.0 * math.pi)
    z = normal_cdf_scalar(high, mean, sd) - normal_cdf_scalar(low, mean, sd)
    t_mean = mean + sd * (phi_alpha - phi_beta) / z
    t_var = sd**2 * (
        1.0
        + (alpha * phi_alpha - beta * phi_beta) / z
        - ((phi_alpha - phi_beta) / z) ** 2
    )
    return t_mean, t_var


# ---------------------------------------------------------------------------
# Ficha de observación sintética
# ---------------------------------------------------------------------------


def save_ficha():
    data = {
        "N_grupo": [1, 2, 3, 4, 5, 6, 7, 8],
        "Hora_llegada": [
            "09:03",
            "09:10",
            "09:18",
            "09:25",
            "09:34",
            "09:40",
            "09:48",
            "09:56",
        ],
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
    df.to_csv(
        os.path.join(BASE_DIR, "data", "ficha_observacion_sintetica.csv"), index=False
    )
    return df


# ---------------------------------------------------------------------------
# Histogramas de validación
# ---------------------------------------------------------------------------


def plot_hist(values, filename, title, xlabel, distribution, params, discrete=False):
    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    if discrete:
        support = params["support"]
        probs = params["probs"]
        bins = np.arange(support.min() - 0.5, support.max() + 1.5, 1)
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.65,
            edgecolor="white",
            label="Muestra",
        )
        markerline, stemlines, baseline = ax.stem(
            support,
            probs,
            linefmt="C3-",
            markerfmt="C3o",
            basefmt=" ",
            label="PMF teórica",
        )
        plt.setp(stemlines, linewidth=1.8)
        plt.setp(markerline, markersize=4.5)
        ax.set_ylabel("Probabilidad")
    else:
        ax.hist(
            values,
            bins=35,
            density=True,
            alpha=0.65,
            edgecolor="white",
            label="Muestra",
        )
        x = np.linspace(min(values), max(values), 500)
        if distribution == "exponential":
            mean = params["mean"]
            y = np.where(x >= 0, np.exp(-x / mean) / mean, 0.0)
        elif distribution == "triangular":
            y = triangular_pdf(x, params["left"], params["mode"], params["right"])
        elif distribution == "uniform":
            low, high = params["low"], params["high"]
            y = np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)
        elif distribution == "truncated_normal":
            y = truncated_normal_pdf(
                x, params["mean"], params["sd"], params["low"], params["high"]
            )
        else:
            raise ValueError(f"Distribución no soportada: {distribution}")
        ax.plot(x, y, color="C3", linewidth=2.2, label="PDF teórica")
        ax.set_ylabel("Densidad")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", filename), dpi=160)
    plt.close()


def validate_distributions(rng):
    n = 10_000
    group_mean, group_var, group_support, group_probs = rounded_triangular_moments(
        4, 10, 18
    )
    normal_reg_mean, normal_reg_var = truncated_normal_moments(5, 1, 3, 7)
    samples = {
        "Interllegadas": exponential_inverse(rng, mean=7.5, size=n),
        "Tamano Grupo": np.maximum(
            1, np.round(triangular_inverse(rng, 4, 10, 18, size=n)).astype(int)
        ),
        "Registro Triangular": triangular_inverse(rng, 3, 5, 7, size=n),
        "Registro Normal Truncada": truncated_normal_box_muller(
            rng, mean=5, sd=1, low=3, high=7, size=n
        ),
        "Navegacion Ida": uniform_inverse(rng, 25, 40, size=n),
        "Permanencia": triangular_inverse(rng, 45, 90, 180, size=n),
    }
    theory = {
        "Interllegadas": {
            "Distribucion": "Exponencial",
            "Media_Teorica": 7.5,
            "Varianza_Teorica": 7.5**2,
        },
        "Tamano Grupo": {
            "Distribucion": "Triangular discreta redondeada",
            "Media_Teorica": group_mean,
            "Varianza_Teorica": group_var,
        },
        "Registro Triangular": {
            "Distribucion": "Triangular",
            "Media_Teorica": triangular_mean(3, 5, 7),
            "Varianza_Teorica": triangular_variance(3, 5, 7),
        },
        "Registro Normal Truncada": {
            "Distribucion": "Normal truncada",
            "Media_Teorica": normal_reg_mean,
            "Varianza_Teorica": normal_reg_var,
        },
        "Navegacion Ida": {
            "Distribucion": "Uniforme",
            "Media_Teorica": (25 + 40) / 2,
            "Varianza_Teorica": (40 - 25) ** 2 / 12,
        },
        "Permanencia": {
            "Distribucion": "Triangular",
            "Media_Teorica": triangular_mean(45, 90, 180),
            "Varianza_Teorica": triangular_variance(45, 90, 180),
        },
    }
    plot_hist(
        samples["Interllegadas"],
        "hist_interllegadas.png",
        "Interllegadas — Exponencial(7.5) con PDF teórica",
        "Minutos",
        "exponential",
        {"mean": 7.5},
    )
    plot_hist(
        samples["Tamano Grupo"],
        "hist_tamano_grupo.png",
        "Tamaño de grupo — Triangular discreta con PMF teórica",
        "Turistas",
        "triangular_discrete",
        {"support": group_support, "probs": group_probs},
        True,
    )
    plot_hist(
        samples["Registro Triangular"],
        "hist_registro.png",
        "Registro — Triangular(3,5,7) con PDF teórica",
        "Minutos",
        "triangular",
        {"left": 3, "mode": 5, "right": 7},
    )
    plot_hist(
        samples["Registro Normal Truncada"],
        "hist_registro_normal.png",
        "Registro — Normal truncada N(5,1²) en [3,7] con PDF teórica",
        "Minutos",
        "truncated_normal",
        {"mean": 5, "sd": 1, "low": 3, "high": 7},
    )
    plot_hist(
        samples["Navegacion Ida"],
        "hist_navegacion.png",
        "Navegación — Uniforme(25,40) con PDF teórica",
        "Minutos",
        "uniform",
        {"low": 25, "high": 40},
    )
    plot_hist(
        samples["Permanencia"],
        "hist_permanencia.png",
        "Permanencia — Triangular(45,90,180) con PDF teórica",
        "Minutos",
        "triangular",
        {"left": 45, "mode": 90, "right": 180},
    )
    df = pd.DataFrame(
        [
            {
                "Variable": name,
                "Distribucion": theory[name]["Distribucion"],
                "Media_Muestral": round(float(np.mean(values)), 4),
                "Media_Teorica": round(float(theory[name]["Media_Teorica"]), 4),
                "Diferencia_Media": round(
                    float(np.mean(values) - theory[name]["Media_Teorica"]), 4
                ),
                "Varianza_Muestral": round(float(np.var(values)), 4),
                "Varianza_Teorica": round(float(theory[name]["Varianza_Teorica"]), 4),
                "Diferencia_Varianza": round(
                    float(np.var(values) - theory[name]["Varianza_Teorica"]), 4
                ),
            }
            for name, values in samples.items()
        ]
    )
    df.to_csv(
        os.path.join(BASE_DIR, "results", "validacion_distribuciones.csv"), index=False
    )
    return df


# ---------------------------------------------------------------------------
# Generación de grupos
# ---------------------------------------------------------------------------


def generar_grupos(rng, duracion, lambda_grupos):
    """Genera grupos turísticos que llegan durante la jornada (hasta t=duracion)."""
    t = 0.0
    grupos = []
    while True:
        t += exponential_inverse(rng, mean=60.0 / lambda_grupos)
        if t > duracion:
            break
        grupos.append(
            {
                "t_llegada": t,
                "tamano": max(1, int(round(triangular_inverse(rng, 4, 10, 18)))),
                "t_registro": triangular_inverse(rng, 3, 5, 7),
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

        ciclo_sin_prep = (
            uniform_inverse(rng, 25, 40)
            + triangular_inverse(rng, 45, 90, 180)
            + uniform_inverse(rng, 25, 40)
        )
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

        cola = cola[len(cargados) :]

    # Turistas no atendidos = los que quedaron en cola al cierre de jornada
    no_atendidos = sum(g["tamano"] for g in cola)

    tiempo_total = duracion * n_embarcaciones  # ventana fija
    espera_prom = float(np.mean(esperas)) if esperas else 0.0
    return {
        "Espera_Promedio": espera_prom,
        "Sistema_Promedio": float(np.mean(sistemas)) if sistemas else 0.0,
        "Longitud_Cola": (lambda_grupos / 60.0) * espera_prom,
        "Utilizacion": ocupado / tiempo_total if tiempo_total else 0.0,
        "Porcentaje_Espera": (
            float(np.mean(np.array(esperas_cola) > 0)) if esperas_cola else 0.0
        ),
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
        {
            "Escenario": "E0 Base",
            "lambda": 8,
            "n_emb": 3,
            "cap": 30,
            "prep": 10,
            "umbral": 24,
        },
        {
            "Escenario": "E1 Más embarcaciones",
            "lambda": 8,
            "n_emb": 4,
            "cap": 30,
            "prep": 10,
            "umbral": 24,
        },
        {
            "Escenario": "E2 Menor preparación",
            "lambda": 8,
            "n_emb": 3,
            "cap": 30,
            "prep": 5,
            "umbral": 24,
        },
        {
            "Escenario": "E3 Mayor capacidad",
            "lambda": 8,
            "n_emb": 3,
            "cap": 40,
            "prep": 10,
            "umbral": 32,
        },
        {
            "Escenario": "E4 Alta demanda",
            "lambda": 10,
            "n_emb": 3,
            "cap": 30,
            "prep": 10,
            "umbral": 24,
        },
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
        row |= {
            str(k): v for k, v in df.mean(numeric_only=True).round(3).to_dict().items()
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(
        os.path.join(BASE_DIR, "results", "resultados_escenarios.csv"), index=False
    )
    return result


# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------


def plot_scenarios(df):
    charts = [
        (
            "Espera_Promedio",
            "escenarios_espera_promedio.png",
            "Tiempo promedio de espera por escenario",
            "Minutos",
        ),
        (
            "Percentil_95_Espera",
            "escenarios_percentil95.png",
            "Percentil 95 de espera por escenario",
            "Minutos",
        ),
        (
            "Utilizacion",
            "escenarios_utilizacion.png",
            "Utilización promedio de embarcaciones",
            "Proporción",
        ),
        (
            "Turistas_No_Atendidos",
            "escenarios_turistas_no_atendidos.png",
            "Turistas no atendidos por escenario",
            "Turistas",
        ),
    ]
    for col, filename, title, ylabel in charts:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=df,
            x="Escenario",
            y=col,
            hue="Escenario",
            legend=False,
            palette="viridis",
        )
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

La simulación base mantiene registro triangular. La distribución Normal truncada
N(5, 1²) en [3, 7] se incluye como validación complementaria exigida por el
trabajo académico, sin afirmar que provenga de datos de campo oficiales.
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
