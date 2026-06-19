"""
geodesics.py — Ecuaciones geodésicas de Schwarzschild como sistema de EDOs.

Las geodésicas se derivan del lagrangiano:
    2L = -f(r)(dt/dλ)² + (1/f(r))(dr/dλ)² + r²(dφ/dλ)²

Las cantidades conservadas son:
    E = f(r) dt/dλ    (energía específica por unidad de masa en reposo)
    L = r² dφ/dλ      (momento angular específico)

Esto reduce el sistema a la ecuación radial:
    (dr/dλ)² = E² - f(r)(ε + L²/r²)

donde ε = 1 (partícula masiva) o ε = 0 (fotón).

Para integrar numéricamente se define el estado:
    y = [t, r, φ, ṙ]   donde ṙ = dr/dλ

y el sistema de primer orden:
    ṫ   = dt/dλ  = E / f(r)
    ṙ   = dr/dλ  (componente del estado)
    φ̇   = dφ/dλ  = L / r²
    r̈   = d²r/dλ² (calculado por diferenciación)

La segunda derivada radial se obtiene diferenciando (ṙ)² respecto a λ:
    2ṙ r̈ = -f'(r)ṙ(ε + L²/r²) - f(r)(2L²/r³)ṙ

Si ṙ ≠ 0, simplificando:
    r̈ = [-f'(r)(ε + L²/r²) + 2L²f(r)/r³] / 2

Con f'(r) = 2/r², esto da:
    r̈ = -ε/r² + L²/r³ - 3L²/r⁴

Verificación: para la esfera de fotones (ε=0, r̈=0):
    L²/r³ = 3L²/r⁴  →  r = 3M  ✓

Para ISCO masiva (ε=1, r̈=0, mínimo de V_eff):
    -1/r² + L²/r³ - 3L²/r⁴ = 0  →  r = 6M con L = 2√3  ✓
"""

import numpy as np
from metric import f, df_dr, esta_dentro_horizonte


def sistema_geodesico(lam, y, E, L, epsilon):
    """
    Sistema de EDOs para geodésicas de Schwarzschild.

    Parámetros:
        lam     : parámetro afín λ (variable independiente)
        y       : estado [t, r, φ, ṙ]
        E       : energía específica (constante de movimiento)
        L       : momento angular específico (constante de movimiento)
        epsilon : 1 (masiva) o 0 (fotón)

    Retorna:
        [dt/dλ, dr/dλ, dφ/dλ, d²r/dλ²]
    """
    t, r, phi, rdot = y

    # Detener si el fotón/partícula cruza el horizonte
    if r <= 2.0 + 1e-8:
        return [0.0, 0.0, 0.0, 0.0]

    fr = f(r)       # f(r) = 1 - 2/r
    dfr = df_dr(r)  # df/dr = 2/r²

    # dt/dλ = E / f(r)  (de la conservación de la energía)
    tdot = E / fr

    # dφ/dλ = L / r²  (de la conservación del momento angular)
    phidot = L / r**2

    # d²r/dλ² por diferenciación de (ṙ)² = E² - f(r)(ε + L²/r²)
    # r̈ = -ε/r² + L²/r³ - 3L²/r⁴
    L2 = L**2
    rddot = -epsilon / r**2 + L2 / r**3 - 3.0 * L2 / r**4

    return [tdot, rdot, phidot, rddot]


def condiciones_iniciales(r0, phi0, E, L, epsilon, rdot0=None, sentido=-1):
    """
    Construye el vector de condiciones iniciales [t0, r0, φ0, ṙ0].

    Si rdot0 es None, se calcula como ±√(E² - V_eff(r0)).
    El parámetro `sentido` controla la dirección radial inicial:
        -1 : hacia el centro (infall)
        +1 : alejándose del centro (outgoing)

    Lanza ValueError si r0 está en una región clásicamente prohibida.

    Retorna:
        (y0, aviso) donde y0 = [t0, r0, φ0, ṙ0] y aviso es None o
        una cadena de advertencia si rdot0 manual no cumple la condición
        de energía E² = V_eff(r0) + rdot0² con tolerancia 1e-6.
    """
    from metric import f as fr_func

    # Verificar que el radio inicial es físico
    if r0 <= 2.0:
        raise ValueError(f"r0 = {r0:.3f} está dentro del horizonte (r ≤ 2M).")

    aviso = None  # Advertencia sobre rdot0 manual (si procede)

    if rdot0 is None:
        # Calcular ṙ0 automáticamente a partir de la condición de energía
        V0 = fr_func(r0) * (epsilon + L**2 / r0**2)
        discriminante = E**2 - V0
        if discriminante < -1e-8:
            raise ValueError(
                f"Región prohibida: E² = {E**2:.4f} < V_eff(r0) = {V0:.4f}. "
                f"Ajuste E o L."
            )
        discriminante = max(discriminante, 0.0)
        rdot0 = sentido * np.sqrt(discriminante)
    else:
        # Validar rdot0 manual: comprobar E² = V_eff(r0) + rdot0²
        V0 = fr_func(r0) * (epsilon + L**2 / r0**2)
        residuo = abs(E**2 - V0 - rdot0**2)
        if residuo > 1e-6:
            aviso = (
                f"AVISO: rdot0 manual no cumple la condicion de energia. "
                f"Residuo = {residuo:.3e}"
            )

    return [0.0, r0, phi0, rdot0], aviso
