"""
presets.py — Casos predefinidos físicamente significativos para la aplicación.

Cada preset es un diccionario con los parámetros necesarios para integrar
una geodésica y una descripción física de lo que representa.

Todas las cantidades están en unidades geométricas (G = c = M = 1).
"""

import numpy as np
from potentials import L_circular, E_circular, potencial_efectivo
from metric import B_CRITICO, R_ISCO, R_FOTON

# ==============================================================================
# Presets para partículas masivas (epsilon = 1)
# ==============================================================================

# ISCO: L = 2√3, E = √(8/9), r0 = 6M
L_ISCO = 2.0 * np.sqrt(3.0)   # ≈ 3.4641
E_ISCO = np.sqrt(8.0 / 9.0)   # ≈ 0.9428

PRESETS_MASIVAS = {
    "Órbita circular estable (r=10M)": {
        "descripcion": (
            "Órbita circular estable a r = 10M. "
            "Es un mínimo del potencial efectivo: ṙ = 0, r̈ = 0."
        ),
        "E": E_circular(10.0),
        "L": L_circular(10.0),
        "r0": 10.0,
        "phi0": 0.0,
        "rdot0": 0.0,
        "sentido": 1,
        "epsilon": 1,
        "lambda_max": 2000.0,
    },
    "ISCO (r=6M)": {
        "descripcion": (
            "Última órbita circular estable (ISCO) a r = 6M. "
            "Con L = 2√3 ≈ 3.464, E = √(8/9) ≈ 0.943. "
            "El mínimo y el máximo del potencial coinciden."
        ),
        "E": E_ISCO,
        "L": L_ISCO,
        "r0": R_ISCO,
        "phi0": 0.0,
        "rdot0": 0.0,
        "sentido": 1,
        "epsilon": 1,
        "lambda_max": 5000.0,
    },
    "Órbita ligada (precesión)": {
        "descripcion": (
            "Órbita ligada con precesión relativista del periastro visible. "
            "Para E=0.975 y L=4.0, la partícula oscila entre r≈6.76M y r≈30.61M. "
            "La trayectoria forma una roseta debido a la curvatura del espacio-tiempo. "
            "Régimen relativista fuerte (p≈11M ≈ 2×ISCO): la fórmula 6π/p no es precisa."
        ),
        "E": 0.975,
        "L": 4.0,
        "r0": 8.0,
        "phi0": 0.0,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 1,
        "lambda_max": 8000.0,
    },
    "Validación precesión campo débil": {
        "descripcion": (
            "Órbita ligada en régimen de campo débil para validar la fórmula 6π/p. "
            "Con r_peri ≈ 40M y r_apo ≈ 60M (p ≈ 48M, e ≈ 0.2), la aproximación "
            "analítica de primer orden es válida (p >> 6M, e < 0.25). "
            "La discrepancia numérica/analítica debe ser pequeña y permite verificar "
            "que el integrador reproduce el exceso de precesión relativista predicho "
            "por la RG en el límite de campo débil."
        ),
        "E": 0.9902,
        "L": 7.16,
        "r0": 50.0,
        "phi0": 0.0,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 1,
        "lambda_max": 15000.0,
    },
    "Órbita inestable (cerca de r=5M)": {
        "descripcion": (
            "Órbita circular inestable exactamente en r = 5M (máximo del potencial). "
            "Cualquier pequeña perturbación causa captura o escape."
        ),
        "E": E_circular(5.0),
        "L": L_circular(5.0),
        "r0": 5.0,
        "phi0": 0.0,
        "rdot0": 0.0,
        "sentido": 1,
        "epsilon": 1,
        "lambda_max": 3000.0,
    },
    "Captura gravitatoria": {
        "descripcion": (
            "Partícula que cae en espiral hacia el horizonte de sucesos. "
            "Momento angular insuficiente para escapar o quedar en órbita."
        ),
        "E": 0.95,
        "L": 2.5,
        "r0": 5.0,
        "phi0": 0.0,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 1,
        "lambda_max": 2000.0,
    },
    "Escape desde órbita baja": {
        "descripcion": (
            "Partícula con energía suficiente para escapar al infinito (E ≥ 1). "
            "Se aleja indefinidamente del agujero negro."
        ),
        "E": 1.05,
        "L": 4.5,
        "r0": 8.0,
        "phi0": 0.0,
        "rdot0": None,
        "sentido": 1,
        "epsilon": 1,
        "lambda_max": 1500.0,
    },
}

# ==============================================================================
# Presets para fotones (epsilon = 0)
# ==============================================================================

# Parámetro de impacto crítico: b_c = 3√3 ≈ 5.196
# Para fotones: E = 1 (normalización), L = b (parámetro de impacto)

PRESETS_FOTONES = {
    "Dispersión (b > b_c)": {
        "descripcion": (
            f"Fotón con b = 6.5 > b_c ≈ {B_CRITICO:.3f}. "
            "El fotón es desviado por el campo gravitatorio pero escapa al infinito."
        ),
        "E": 1.0,
        "L": 6.5,
        "r0": 50.0,
        "phi0": np.pi,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 0,
        "lambda_max": 700.0,
    },
    "Captura (b < b_c)": {
        "descripcion": (
            f"Fotón con b = 4.0 < b_c ≈ {B_CRITICO:.3f}. "
            "El campo gravitatorio es suficientemente fuerte para capturar el fotón."
        ),
        "E": 1.0,
        "L": 4.0,
        "r0": 30.0,
        "phi0": np.pi,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 0,
        "lambda_max": 200.0,
    },
    "Trayectoria crítica (b ≈ b_c)": {
        "descripcion": (
            f"Fotón con b ≈ b_c = 3√3 ≈ {B_CRITICO:.3f}. "
            "El fotón orbita aproximadamente en la esfera de fotones (r = 3M) "
            "durante muchas vueltas antes de escapar o ser capturado."
        ),
        "E": 1.0,
        "L": B_CRITICO * 1.0001,
        "r0": 30.0,
        "phi0": np.pi,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 0,
        "lambda_max": 500.0,
    },
    "Esfera de fotones (r=3M)": {
        "descripcion": (
            "Fotón en órbita circular inestable exactamente en la esfera de fotones r = 3M. "
            "Cualquier perturbación hace que escape o sea capturado."
        ),
        "E": 1.0,
        "L": B_CRITICO,
        "r0": R_FOTON,
        "phi0": 0.0,
        "rdot0": 0.0,
        "sentido": 1,
        "epsilon": 0,
        "lambda_max": 300.0,
    },
    "Comparación cerca de b_c": {
        "descripcion": (
            "Tres fotones con b ligeramente diferente al crítico. "
            "Muestra la sensibilidad a b_c (ver modo comparador)."
        ),
        "E": 1.0,
        "L": B_CRITICO,
        "r0": 30.0,
        "phi0": np.pi,
        "rdot0": None,
        "sentido": -1,
        "epsilon": 0,
        "lambda_max": 400.0,
    },
}

# ==============================================================================
# Trayectorias para el comparador (parámetro de impacto)
# ==============================================================================

def trayectorias_deflexion_foton():
    """
    Retorna configs para comparar el ángulo de deflexión gravitacional de
    fotones con distintos parámetros de impacto b > b_c (dispersión).

    Todos los fotones parten de r0 = 100 M con phi0 = π (entrando desde
    radio grande) y escapan tras pasar su periapsis. La variación de b
    permite verificar la transición de campo fuerte a campo débil.
    """
    configs = []
    valores_b = [
        (6.0,  "#FF4444", "b = 6.0 M  (campo fuerte)"),
        (6.5,  "#FF8844", "b = 6.5 M"),
        (7.0,  "#FFFF44", "b = 7.0 M"),
        (8.0,  "#88FF44", "b = 8.0 M"),
        (10.0, "#44AAFF", "b = 10.0 M"),
        (15.0, "#FF44FF", "b = 15.0 M"),
        (25.0, "#44FFFF", "b = 25.0 M (campo debil)"),
    ]
    for b, color, etiqueta in valores_b:
        configs.append({
            "E": 1.0,
            "L": b,
            "r0": 100.0,
            "phi0": np.pi,
            "rdot0": None,
            "sentido": -1,
            "epsilon": 0,
            "lambda_max": 1200.0,
            "etiqueta": etiqueta,
            "color": color,
        })
    return configs


def trayectorias_comparacion_bc():
    """
    Retorna una lista de configuraciones para comparar trayectorias de fotones
    con b < b_c, b = b_c y b > b_c.
    """
    configs = []
    valores_b = [
        (B_CRITICO * 0.85, "b = 0.85 b_c  (captura)",   "#FF4444"),
        (B_CRITICO * 0.95, "b = 0.95 b_c  (captura)",   "#FF8844"),
        (B_CRITICO,        "b = b_c        (crítico)",  "#FFFF44"),
        (B_CRITICO * 1.05, "b = 1.05 b_c  (escape)",    "#88FF44"),
        (B_CRITICO * 1.20, "b = 1.20 b_c  (escape)",    "#44AAFF"),
    ]
    for b, etiqueta, color in valores_b:
        configs.append({
            "E": 1.0,
            "L": b,
            "r0": 40.0,
            "phi0": np.pi,
            "rdot0": None,
            "sentido": -1,
            "epsilon": 0,
            "lambda_max": 400.0,
            "etiqueta": etiqueta,
            "color": color,
        })
    return configs


# ==============================================================================
# Verificación automática de presets
# ==============================================================================

def verificar_todos_los_presets():
    """
    Comprueba que cada preset tiene E² >= V_eff(r0), condición necesaria para
    que la partícula pueda estar en r0 con el momento angular indicado.
    Imprime un resumen de cuáles pasan y cuáles fallan.
    """
    import sys
    enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'

    def _print(texto):
        try:
            print(texto)
        except UnicodeEncodeError:
            print(texto.encode(enc, errors='replace').decode(enc, errors='replace'))

    grupos = [
        ("MASIVAS", PRESETS_MASIVAS),
        ("FOTONES", PRESETS_FOTONES),
    ]
    n_ok = 0
    n_fallo = 0

    _print("\n=== Verificacion de condiciones iniciales de presets ===")
    for nombre_grupo, presets in grupos:
        _print(f"\n  [{nombre_grupo}]")
        for nombre, p in presets.items():
            E   = p["E"]
            L   = p["L"]
            r0  = p["r0"]
            eps = p["epsilon"]
            E2     = E ** 2
            V      = potencial_efectivo(r0, L, eps)
            margen = E2 - V
            valido = margen >= -1e-9
            estado = "OK   " if valido else "FALLA"
            linea = (
                f"    {estado}  {nombre:<45s}"
                f"  E2={E2:.5f}  V_eff(r0)={V:.5f}  d={margen:+.5f}"
            )
            _print(linea)
            if valido:
                n_ok += 1
            else:
                n_fallo += 1

    _print(f"\n  Resultado: {n_ok} OK, {n_fallo} fallo(s)")
    _print("=" * 55)


if __name__ == "__main__":
    verificar_todos_los_presets()
