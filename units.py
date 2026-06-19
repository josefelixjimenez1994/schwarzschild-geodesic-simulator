"""
units.py — Conversiones entre unidades geométricas y unidades físicas.

En unidades geométricas (G = c = 1), la masa M tiene dimensiones de longitud.
La escala física es:

    r_físico [km] = r_geométrico [M] × (G·M / c²) [km por M]

donde G·M_sol/c² ≈ 1.47627 km es el "radio geométrico" de una masa solar.

La simulación corre siempre en M = 1; esta clase solo traduce resultados
a unidades físicas para mostrarlos en pantalla.
"""

import numpy as np

# Constantes en SI
_G   = 6.67430e-11    # m³ kg⁻¹ s⁻²
_c   = 2.99792e8      # m s⁻¹
_M_sol_kg = 1.98892e30  # kg

# Escala de longitud de una masa solar en km: G·M_sol/c²
_GM_SOL_KM = _G * _M_sol_kg / _c**2 / 1e3   # ≈ 1.47627 km

# Escala de tiempo de una masa solar en ms: G·M_sol/c³
_GM_SOL_MS = _G * _M_sol_kg / _c**3 * 1e3   # ≈ 4.92549 µs → en ms


# ==============================================================================
# Masas predefinidas (en masas solares)
# ==============================================================================

MASAS_PREDEFINIDAS = {
    "Estelar — 10 M☉":              10.0,
    "Estelar masivo — 30 M☉":       30.0,
    "N. estelar mínima — 1.4 M☉":   1.4,
    "Intermedio — 1 000 M☉":        1.0e3,
    "Supermasivo — 1×10⁶ M☉":       1.0e6,
    "Sgr A* — 4.1×10⁶ M☉":         4.1e6,
    "M87* — 6.5×10⁹ M☉":           6.5e9,
}


# ==============================================================================
# Funciones de conversión
# ==============================================================================

def r_a_km(r_geo, M_solar):
    """Convierte radio en unidades de M a km."""
    return r_geo * M_solar * _GM_SOL_KM


def t_a_ms(t_geo, M_solar):
    """Convierte tiempo en unidades de M a milisegundos."""
    return t_geo * M_solar * _GM_SOL_MS


def radios_criticos_km(M_solar):
    """Devuelve los tres radios de Schwarzschild en km."""
    return {
        "horizonte": r_a_km(2.0, M_solar),
        "foton":     r_a_km(3.0, M_solar),
        "isco":      r_a_km(6.0, M_solar),
    }


def _fmt(valor_km):
    """Formatea un valor en km eligiendo la unidad más legible."""
    if valor_km >= 1.0e9:
        return f"{valor_km/1.496e8:.3g} UA"     # unidades astronómicas
    if valor_km >= 1.0e6:
        return f"{valor_km/1.496e8:.3g} UA  ({valor_km:.3g} km)"
    if valor_km >= 1.0:
        return f"{valor_km:.4g} km"
    return f"{valor_km*1e3:.4g} m"


def texto_escala(M_solar):
    """
    Genera un bloque de texto con los radios críticos en unidades físicas.
    Listo para mostrar en un QTextEdit o QLabel.
    """
    r = radios_criticos_km(M_solar)
    escala_km = r_a_km(1.0, M_solar)

    lineas = [
        f"M = {M_solar:.4g} M☉",
        f"  1 M  =  {escala_km:.4g} km",
        "",
        f"  R_H   = 2M  =  {_fmt(r['horizonte'])}",
        f"  R_ph  = 3M  =  {_fmt(r['foton'])}",
        f"  ISCO  = 6M  =  {_fmt(r['isco'])}",
    ]
    return "\n".join(lineas)
