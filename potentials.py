"""
potentials.py — Potenciales efectivos para geodésicas en Schwarzschild.

El movimiento radial se reduce a un problema unidimensional con potencial efectivo.
La ecuación radial es:

    (dr/dλ)² = E² - V_eff(r)

donde V_eff depende del tipo de partícula:

    Partículas masivas (ε = 1):
        V_eff(r) = (1 - 2/r)(1 + L²/r²)

    Fotones (ε = 0):
        V_eff(r) = (1 - 2/r) L²/r²

El movimiento es posible donde E² ≥ V_eff(r) (región clásicamente permitida).
Los puntos de retorno son las raíces de E² = V_eff(r).
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from metric import f, df_dr, R_HORIZONTE, R_FOTON, R_ISCO, B_CRITICO


# ==============================================================================
# Potencial efectivo y su derivada
# ==============================================================================

def potencial_efectivo(r, L, epsilon):
    """
    Potencial efectivo V_eff(r, L).

    Parámetros:
        r       : radio (escalar o array)
        L       : momento angular específico
        epsilon : 1 (partícula masiva) o 0 (fotón)

    Retorna:
        V_eff = f(r) * (ε + L²/r²)
    """
    return f(r) * (epsilon + L**2 / r**2)


def dVeff_dr(r, L, epsilon):
    """
    Derivada del potencial efectivo respecto a r.

    dV_eff/dr = f'(r)(ε + L²/r²) + f(r)(-2L²/r³)
    """
    return df_dr(r) * (epsilon + L**2 / r**2) + f(r) * (-2.0 * L**2 / r**3)


# ==============================================================================
# Radios críticos y condiciones de órbita circular
# ==============================================================================

def L_circular(r, epsilon=1):
    """
    Momento angular necesario para una órbita circular de radio r (partícula masiva).

    Para ε = 1: dV_eff/dr = 0 implica L² = r² / (r - 3)
    Solo existe para r > 3M (fuera de la esfera de fotones).
    """
    if epsilon == 0:
        return None
    if np.isscalar(r):
        if r <= 3.0:
            return None
        return np.sqrt(r**2 / (r - 3.0))
    else:
        resultado = np.where(r > 3.0, np.sqrt(r**2 / np.maximum(r - 3.0, 1e-10)), np.nan)
        return resultado


def E_circular(r, epsilon=1):
    """
    Energía específica para una órbita circular de radio r (partícula masiva).

    Para ε = 1: E = (r - 2) / √(r(r - 3))
    Solo existe para r > 3M.
    """
    if epsilon == 0:
        return None
    if np.isscalar(r):
        if r <= 3.0:
            return None
        return (r - 2.0) / np.sqrt(r * (r - 3.0))
    else:
        mascara = r > 3.0
        resultado = np.where(
            mascara,
            (r - 2.0) / np.sqrt(r * np.maximum(r - 3.0, 1e-10)),
            np.nan
        )
        return resultado


def encontrar_radios_circulares(L, epsilon=1):
    """
    Encuentra los radios de las órbitas circulares (extremos de V_eff).

    Para partículas masivas, resuelve dV_eff/dr = 0:
        r² - L²r + 3L² = 0

    Retorna:
        (r_estable, r_inestable) o None si no existen soluciones reales.
        r_estable > r_inestable (el mínimo del potencial es la órbita estable).
    """
    if epsilon == 0:
        # Para fotones, el único radio circular es la esfera de fotones r = 3M
        return (None, R_FOTON)

    discriminante = L**4 - 12.0 * L**2
    if discriminante < 0:
        # No hay órbitas circulares para L < 2√3 (ISCO)
        return None

    sqrt_D = np.sqrt(discriminante)
    r_estable = (L**2 + sqrt_D) / 2.0     # Mínimo del potencial
    r_inestable = (L**2 - sqrt_D) / 2.0   # Máximo del potencial

    if r_inestable <= R_HORIZONTE:
        return (r_estable, None)

    return (r_estable, r_inestable)


# ==============================================================================
# Puntos de retorno y clasificación de trayectorias
# ==============================================================================

def encontrar_puntos_retorno(E, L, epsilon, r_min=2.01, r_max=500.0, n_puntos=10000):
    """
    Encuentra los puntos de retorno donde E² = V_eff(r) (cambio de signo de ṙ).

    Usa búsqueda de raíces con brentq en intervalos donde V_eff - E² cambia de signo.

    Retorna:
        Lista ordenada de radios donde el movimiento radial se detiene.
    """
    E2 = E**2
    r_arr = np.linspace(r_min, r_max, n_puntos)

    # Evitar singularidades cercanas al horizonte
    r_arr = r_arr[r_arr > R_HORIZONTE + 0.001]

    try:
        V_arr = potencial_efectivo(r_arr, L, epsilon)
    except Exception:
        return []

    diferencia = V_arr - E2
    puntos = []

    for i in range(len(r_arr) - 1):
        if np.isfinite(diferencia[i]) and np.isfinite(diferencia[i + 1]):
            if diferencia[i] * diferencia[i + 1] < 0:
                try:
                    r_ret = brentq(
                        lambda r: potencial_efectivo(r, L, epsilon) - E2,
                        r_arr[i], r_arr[i + 1],
                        xtol=1e-10, maxiter=100
                    )
                    puntos.append(r_ret)
                except Exception:
                    pass

    return sorted(puntos)


def clasificar_trayectoria_teorica(E, L, epsilon, r0=None):
    """
    Clasifica el tipo de trayectoria a partir de E y L, sin integrar.

    Distingue los casos: ISCO, órbita circular estable, órbita circular
    inestable, esfera de fotones, órbita ligada, captura y escape.

    Retorna una cadena descriptiva del tipo de órbita.
    """
    E2 = E**2
    b = abs(L / E) if abs(E) > 1e-10 else np.inf

    if epsilon == 0:  # Fotón
        if abs(b - B_CRITICO) < 0.05 * B_CRITICO:
            # La esfera de fotones es la órbita circular inestable de los fotones
            return f"Esfera de fotones (órbita circular inestable, b ≈ b_c = {B_CRITICO:.3f})"
        elif b < B_CRITICO:
            return "Captura (b < b_c)"
        else:
            return "Dispersión (b > b_c)"

    # --- Partícula masiva ---

    # Valores en la ISCO: L_ISCO = 2√3, E_ISCO = √(8/9), r = 6M
    L_ISCO = 2.0 * np.sqrt(3.0)  # ≈ 3.4641
    E_ISCO = np.sqrt(8.0 / 9.0)  # ≈ 0.9428

    # Detectar ISCO: L y E próximos a los valores críticos (tolerancia 2 %)
    if abs(L - L_ISCO) < 0.02 * L_ISCO and abs(E - E_ISCO) < 0.02 * E_ISCO:
        return "ISCO (r = 6M, orbita circular marginalmente estable)"

    radios_circ = encontrar_radios_circulares(L, epsilon)

    # Detectar órbitas circulares exactas comparando E² con V_eff en los extremos
    if radios_circ:
        r_est, r_inest = radios_circ
        if r_est is not None:
            E_est = E_circular(r_est)
            if E_est is not None and abs(E2 - E_est**2) < 1e-4 * max(E2, 1e-10):
                return f"Orbita circular estable (r = {r_est:.3f}M)"
        if r_inest is not None:
            E_inest = E_circular(r_inest)
            if E_inest is not None and abs(E2 - E_inest**2) < 1e-4 * max(E2, 1e-10):
                return f"Orbita circular inestable (r = {r_inest:.3f}M)"

    puntos = encontrar_puntos_retorno(E, L, epsilon)

    if len(puntos) == 0:
        if E2 >= 1.0:
            return "Escape desde el infinito"
        else:
            return "Captura gravitatoria"
    elif len(puntos) == 1:
        if E2 >= 1.0:
            return "Dispersion (orbita hiperbolica)"
        else:
            return "Captura gravitatoria (orbita parabolica)"
    elif len(puntos) >= 2:
        # Con tres raíces (masiva, E<1) la raíz más interior pertenece a una
        # región separada por la barrera; la órbita ligada exterior usa las dos
        # últimas raíces.
        if epsilon == 1 and E2 < 1.0 and len(puntos) >= 3:
            r_min_ret = puntos[-2]
            r_max_ret = puntos[-1]
        else:
            r_min_ret = puntos[0]
            r_max_ret = puntos[-1]

        # Para E²≥1 (partícula masiva) las dos raíces son las paredes de una
        # barrera potencial, no los extremos de una órbita ligada. V_eff→1<E²
        # en el infinito, así que la región exterior siempre se extiende al
        # infinito. Usamos r0 para saber en qué rama física está la partícula.
        if epsilon == 1 and E2 >= 1.0:
            if r0 is not None and np.isfinite(r0):
                if r0 > r_max_ret:
                    return "Escape relativista (E ≥ 1, barrera potencial no cruzada)"
                elif r0 < r_min_ret:
                    return "Captura gravitatoria (E ≥ 1, región interior a la barrera)"
            return "Escape o Captura (E ≥ 1, barrera potencial)"

        # Comprobar si los puntos de retorno son casi iguales → órbita cuasi-circular
        if radios_circ:
            r_est, _ = radios_circ
            if r_est and abs(r_min_ret - r_max_ret) < 0.01 * r_est:
                return f"Orbita circular estable (r ≈ {r_est:.3f}M)"
        return f"Orbita ligada (r_min = {r_min_ret:.3f}M, r_max = {r_max_ret:.3f}M)"

    return "Indeterminada"


def regiones_permitidas(E, L, epsilon, r_arr):
    """
    Devuelve una máscara booleana indicando dónde E² ≥ V_eff(r).
    Usado para colorear el gráfico del potencial.
    """
    V = potencial_efectivo(r_arr, L, epsilon)
    return E**2 >= V
