"""
metric.py — Métrica de Schwarzschild en unidades geométricas (G = c = M = 1).

La métrica de Schwarzschild es la solución exacta a las ecuaciones de Einstein
en el vacío para un cuerpo esférico no rotante de masa M:

    ds² = -f(r) dt² + dr²/f(r) + r² dΩ²

donde f(r) = 1 - 2M/r. Con G = c = M = 1, esto se simplifica a f(r) = 1 - 2/r.
"""

# ==============================================================================
# Radios característicos (en unidades de M, con M = 1)
# ==============================================================================

# Horizonte de sucesos: f(r_H) = 0 → r_H = 2M = 2
R_HORIZONTE = 2.0

# Esfera de fotones: órbita circular inestable de fotones r_ph = 3M = 3
R_FOTON = 3.0

# Última órbita circular estable (ISCO): r_ISCO = 6M = 6
R_ISCO = 6.0

# Parámetro de impacto crítico para fotones: b_c = 3√3 M
import numpy as np
B_CRITICO = 3.0 * np.sqrt(3.0)


def f(r):
    """
    Función de Schwarzschild: f(r) = 1 - 2M/r, con M = 1.

    Aparece en el componente temporal de la métrica: g_tt = -f(r).
    Se anula en el horizonte (r = 2M) y tiende a 1 en el infinito.
    """
    return 1.0 - 2.0 / r


def df_dr(r):
    """
    Derivada de la función de Schwarzschild respecto a r: df/dr = 2M/r² = 2/r².
    Necesaria para calcular las fuerzas geodésicas.
    """
    return 2.0 / r**2


def esta_dentro_horizonte(r, margen=1e-5):
    """Devuelve True si r ha cruzado o está muy cerca del horizonte."""
    return r <= R_HORIZONTE + margen


def dilatacion_temporal(r):
    """
    Factor de dilatación temporal gravitatoria: dτ/dt = √(1 - 2M/r).
    En el horizonte vale 0; en el infinito vale 1.
    """
    val = f(r)
    return np.sqrt(np.maximum(val, 0.0))
