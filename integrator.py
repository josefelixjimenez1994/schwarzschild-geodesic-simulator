"""
integrator.py — Integrador numérico para geodésicas de Schwarzschild.

Usa scipy.integrate.solve_ivp con el método DOP853 (Dormand-Prince de orden 8),
que es de alta precisión y adecuado para trayectorias con fuerte curvatura.

Se definen dos eventos de parada:
    - Captura:  r alcanza el horizonte (r → 2M⁺)
    - Escape:   r supera un radio máximo configurable

Estructura del resultado (diccionario):
    'lam'   : valores del parámetro afín λ
    't'     : tiempo coordenado t(λ)
    'r'     : coordenada radial r(λ)
    'phi'   : ángulo azimutal φ(λ)
    'rdot'  : velocidad radial ṙ(λ)
    'tipo'  : 'captura', 'escape', 'ligada', 'error'
    'exito' : True si la integración terminó correctamente
    'E'     : energía usada
    'L'     : momento angular usado
    'epsilon': tipo de partícula
"""

import numpy as np
from scipy.integrate import solve_ivp
from geodesics import sistema_geodesico, condiciones_iniciales
from metric import R_HORIZONTE
from potentials import potencial_efectivo, clasificar_trayectoria_teorica


def integrar_geodesica(
    E, L, epsilon,
    r0, phi0=0.0, rdot0=None, sentido=-1,
    lambda_max=3000.0, tolerancia=1e-10,
    r_max=300.0, metodo='DOP853',
    n_puntos_max=20000
):
    """
    Integra una geodésica en la métrica de Schwarzschild.

    Parámetros:
        E           : energía específica
        L           : momento angular específico
        epsilon     : 1 (masiva) o 0 (fotón)
        r0          : radio inicial (debe ser > 2M)
        phi0        : ángulo azimutal inicial (radianes), por defecto 0
        rdot0       : velocidad radial inicial; None = auto (calculado de V_eff)
        sentido     : dirección radial si rdot0=None: -1 (hacia adentro), +1 (hacia afuera)
        lambda_max  : parámetro afín máximo de integración
        tolerancia  : tolerancia relativa de solve_ivp (la absoluta es 1e-3 × tolerancia)
        r_max       : radio de detención por escape
        metodo      : método numérico ('DOP853' recomendado, 'RK45' alternativo)
        n_puntos_max: número máximo de puntos de salida

    Retorna:
        dict con claves: 'lam', 't', 'r', 'phi', 'rdot', 'tipo', 'exito', 'E', 'L', 'epsilon'
    """
    # --- Condiciones iniciales ---
    # aviso_rdot0 captura la advertencia sobre rdot0 manual si no cumple E²=Veff+ṙ²
    aviso_rdot0 = None
    try:
        y0, aviso_rdot0 = condiciones_iniciales(r0, phi0, E, L, epsilon, rdot0, sentido)
    except ValueError as e:
        return _resultado_error(str(e), r0, phi0, E, L, epsilon)

    # --- Definición de eventos de parada ---

    def evento_horizonte(lam, y, E, L, epsilon):
        """Detener cuando r llega al horizonte."""
        return y[1] - (R_HORIZONTE + 1e-5)
    evento_horizonte.terminal = True
    evento_horizonte.direction = -1

    def evento_escape(lam, y, E, L, epsilon):
        """Detener cuando r supera r_max."""
        return y[1] - r_max
    evento_escape.terminal = True
    evento_escape.direction = 1

    # --- Puntos de evaluación uniformemente espaciados ---
    t_eval = np.linspace(0, lambda_max, n_puntos_max)
    paso_max = lambda_max / n_puntos_max

    # --- Integración ---
    try:
        sol = solve_ivp(
            fun=sistema_geodesico,
            t_span=(0.0, lambda_max),
            y0=y0,
            method=metodo,
            args=(E, L, epsilon),
            rtol=tolerancia,
            atol=tolerancia * 1e-3,
            events=[evento_horizonte, evento_escape],
            dense_output=False,
            max_step=paso_max,
            t_eval=t_eval
        )

        lam_arr  = sol.t
        t_arr    = sol.y[0]
        r_arr    = sol.y[1]
        phi_arr  = sol.y[2]
        rdot_arr = sol.y[3]

        # Determinar tipo de trayectoria por evento activado
        if len(sol.t_events[0]) > 0:
            tipo = 'captura'
        elif len(sol.t_events[1]) > 0:
            tipo = 'escape'
        else:
            tipo = 'ligada'

        # Estadísticas de la trayectoria
        r_min = float(np.min(r_arr))
        r_max_traj = float(np.max(r_arr))

        return {
            'lam': lam_arr,
            't': t_arr,
            'r': r_arr,
            'phi': phi_arr,
            'rdot': rdot_arr,
            'tipo': tipo,
            'exito': sol.success,
            'mensaje': sol.message,
            'r_min': r_min,
            'r_max': r_max_traj,
            'E': E,
            'L': L,
            'epsilon': epsilon,
            'aviso_rdot0': aviso_rdot0,  # None si rdot0 era automático o válido
        }

    except Exception as e:
        err = _resultado_error(str(e), r0, phi0, E, L, epsilon)
        err['aviso_rdot0'] = aviso_rdot0  # Preservar advertencia incluso en error tardío
        return err


def _resultado_error(mensaje, r0, phi0, E, L, epsilon):
    """Construye un resultado de error para devolver cuando falla la integración."""
    return {
        'lam': np.array([0.0]),
        't': np.array([0.0]),
        'r': np.array([r0]),
        'phi': np.array([phi0]),
        'rdot': np.array([0.0]),
        'tipo': 'error',
        'exito': False,
        'mensaje': mensaje,
        'r_min': r0,
        'r_max': r0,
        'E': E,
        'L': L,
        'epsilon': epsilon,
        'aviso_rdot0': None,  # Sin advertencia de rdot0 en errores de condiciones iniciales
    }


# ==============================================================================
# Mejora 2: Validador de conservación de energía
# ==============================================================================

def residuo_conservacion(resultado):
    """
    Calcula el residuo de conservación de energía a lo largo de la trayectoria:

        δ(λ) = |E² - V_eff(r(λ)) - ṙ(λ)²|

    Este valor debe ser ≈ 0 en toda la trayectoria.
    Su magnitud valida la precisión del integrador DOP853.

    Retorna (lam, delta) como arrays de igual longitud.
    """
    r    = resultado['r']
    rdot = resultado['rdot']
    E    = resultado['E']
    L    = resultado['L']
    eps  = resultado['epsilon']
    lam  = resultado['lam']

    V = potencial_efectivo(r, L, eps)
    delta = np.abs(E**2 - V - rdot**2)
    return lam, delta


# ==============================================================================
# Mejora 3: Precesión del perihelio
# ==============================================================================

def calcular_precesion_perihelio(resultado):
    """
    Detecta periapsis sucesivos en una órbita ligada y cuantifica la
    precesión del perihelio.

    Método numérico:
        Los periapsis se detectan cuando ṙ cambia de negativo a positivo
        (punto de mínimo acercamiento al agujero negro).

    Fórmula analítica de primer orden (Schwarzschild, M=1):
        Δφ_exceso ≈ 6π / p    [rad/órbita]

    donde p = a(1-e²) = 2 r_peri r_apo / (r_peri + r_apo) es el semi-latus rectum.

    Retorna:
        dict con resultados, o None si no se detectaron suficientes periapsis.
    """
    if resultado['tipo'] not in ('ligada',) or not resultado['exito']:
        return None
    if resultado.get('epsilon', 1) != 1:
        return None

    r    = resultado['r']
    phi  = resultado['phi']
    rdot = resultado['rdot']

    # --- Órbita circular: r_max - r_min < umbral → precesión no definida ---
    r_min_traj = resultado.get('r_min', float(np.min(r)))
    r_max_traj = resultado.get('r_max', float(np.max(r)))
    r_medio    = (r_min_traj + r_max_traj) / 2.0
    if r_medio > 1e-10 and (r_max_traj - r_min_traj) / r_medio < 1e-4:
        return None
    if (r_max_traj - r_min_traj) < 1e-3:
        return None

    # --- Detección de periapsis: ṙ cambia de − a + ---
    # Usar interpolación lineal para mayor precisión en el índice
    idx_peri = []
    for i in range(1, len(rdot)):
        if rdot[i - 1] < 0.0 and rdot[i] >= 0.0:
            # Interpolación entre i-1 e i
            frac = -rdot[i - 1] / (rdot[i] - rdot[i - 1])
            idx_peri.append(i - 1 + frac)

    if len(idx_peri) < 2:
        return None

    # --- Interpolación de r y φ en los periapsis ---
    idx_arr = np.array(idx_peri)
    idx_int = idx_arr.astype(int)
    idx_frac = idx_arr - idx_int

    r_peri_arr   = r[idx_int]   + idx_frac * (r[np.minimum(idx_int + 1, len(r) - 1)]   - r[idx_int])
    phi_peri_arr = phi[idx_int] + idx_frac * (phi[np.minimum(idx_int + 1, len(phi) - 1)] - phi[idx_int])

    # --- Δφ entre periapsis consecutivos ---
    dphi_lista = np.diff(phi_peri_arr)
    dphi_medio = float(np.mean(dphi_lista))

    # --- Radio en apoapsis (entre periapsis consecutivos) ---
    r_apo_lista = []
    idx_int_exact = [int(round(i)) for i in idx_peri]
    for k in range(len(idx_int_exact) - 1):
        i0, i1 = idx_int_exact[k], idx_int_exact[k + 1]
        if i1 > i0:
            r_apo_lista.append(float(np.max(r[i0:i1])))
    r_apo_medio = float(np.mean(r_apo_lista)) if r_apo_lista else float(np.max(r))

    r_peri_medio = float(np.mean(r_peri_arr))

    # --- Semi-latus rectum y excentricidad ---
    p = 2.0 * r_peri_medio * r_apo_medio / (r_peri_medio + r_apo_medio)
    a = (r_peri_medio + r_apo_medio) / 2.0
    e = (r_apo_medio - r_peri_medio) / (r_apo_medio + r_peri_medio)

    # --- Comparación numérico vs analítico ---
    exceso_numerico   = dphi_medio - 2.0 * np.pi
    exceso_analitico  = 6.0 * np.pi / p          # primer orden, M=1

    return {
        'n_periapsis':       len(idx_peri),
        'dphi_medio_rad':    dphi_medio,
        'exceso_num_rad':    exceso_numerico,
        'exceso_anal_rad':   exceso_analitico,
        'r_peri':            r_peri_medio,
        'r_apo':             r_apo_medio,
        'semi_latus_rectum': p,
        'semi_eje_mayor':    a,
        'excentricidad':     e,
    }


# ==============================================================================
# Mejora 6: Tiempo propio acumulado
# ==============================================================================

def calcular_tiempo_propio(resultado):
    """
    Calcula el tiempo propio total τ para una partícula masiva (ε=1).

    Para partículas masivas con el afín λ normalizado como tiempo propio:
        dτ/dλ = 1  →  τ_total = λ_final

    Esta igualdad se cumple exactamente cuando la condición de normalización
    g_μν u^μ u^ν = -1 se satisface en las condiciones iniciales.

    La diferencia τ_total vs t_coordenado ilustra la dilatación temporal
    integrada a lo largo de toda la trayectoria.

    Retorna None para fotones (no tienen tiempo propio definido).
    """
    if resultado['epsilon'] != 1:
        return None
    if not resultado['exito'] or len(resultado['lam']) < 2:
        return None

    tau   = float(resultado['lam'][-1])          # τ = λ para partículas masivas
    t_coo = float(resultado['t'][-1])            # tiempo coordenado acumulado

    ratio = tau / t_coo if t_coo > 1e-10 else 0.0

    return {
        'tau':   tau,
        't_coo': t_coo,
        'ratio': ratio,                          # τ/t: siempre ≤ 1 cerca del horizonte
        'atraso': t_coo - tau,                   # diferencia t - τ (siempre ≥ 0)
    }


# ==============================================================================
# Ángulo de deflexión gravitacional (geodésicas nulas de dispersión)
# ==============================================================================

def calcular_angulo_deflexion(resultado):
    """
    Calcula el ángulo de deflexión gravitacional para fotones de dispersión.

    Condiciones de aplicabilidad:
        - epsilon == 0  (fotón)
        - tipo == 'escape'  (la trayectoria escapa al infinito)
        - existe un periapsis claro (ṙ cambia de − a +)

    Fórmulas:
        Δφ_total = |φ_final − φ_inicial|
        α        = Δφ_total − π
        α_WF     = 4M/b = 4/b   (campo débil, G=c=M=1, primer orden)

    Corrección asintótica de primer orden (aproximación de espacio plano):
        Si la simulación empieza en r0 y termina en r_f (no en ∞), la
        diferencia con el ángulo asintótico real es:
            corr ≈ π − arccos(b/r0) − arccos(b/r_f)
        El valor corregido  α_corr = α + corr  mejora la comparación con
        la fórmula analítica sin necesidad de simular desde r = ∞.

    Retorna dict con todos los valores, o None si la trayectoria no aplica.
    """
    if resultado.get('epsilon', 1) != 0:
        return None
    if resultado['tipo'] != 'escape':
        return None
    if not resultado['exito']:
        return None

    rdot = resultado['rdot']
    r    = resultado['r']
    phi  = resultado['phi']
    E    = resultado['E']
    L    = resultado['L']

    # Verificar periapsis: ṙ debe cambiar de − a + exactamente una vez
    idx_peri = -1
    for i in range(1, len(rdot)):
        if rdot[i - 1] < 0.0 and rdot[i] >= 0.0:
            idx_peri = i
            break

    if idx_peri < 0:
        return None

    # r_min con interpolación lineal en el cruce de signo
    frac = -rdot[idx_peri - 1] / (rdot[idx_peri] - rdot[idx_peri - 1])
    r_min_val = float(r[idx_peri - 1] + frac * (r[idx_peri] - r[idx_peri - 1]))

    # Ángulo barrido durante la simulación
    delta_phi_sim = abs(float(phi[-1]) - float(phi[0]))
    alpha_rad     = delta_phi_sim - np.pi
    alpha_deg     = alpha_rad * 180.0 / np.pi

    # Parámetro de impacto
    b = abs(L / E) if abs(E) > 1e-10 else float('inf')

    # Corrección asintótica de espacio plano (radio finito → ∞)
    r0_val  = float(r[0])
    r_f_val = float(r[-1])
    if b < r0_val and b < r_f_val:
        corr = np.pi - np.arccos(min(b / r0_val, 1.0)) - np.arccos(min(b / r_f_val, 1.0))
        alpha_corr_rad = alpha_rad + corr
        alpha_corr_deg = alpha_corr_rad * 180.0 / np.pi
    else:
        corr           = float('nan')
        alpha_corr_rad = float('nan')
        alpha_corr_deg = float('nan')

    # Fórmula analítica de campo débil
    alpha_WF_rad = 4.0 / b if b > 0 else float('nan')
    alpha_WF_deg = alpha_WF_rad * 180.0 / np.pi if np.isfinite(alpha_WF_rad) else float('nan')

    return {
        'delta_phi_sim':   delta_phi_sim,
        'alpha_rad':       alpha_rad,
        'alpha_deg':       alpha_deg,
        'alpha_corr_rad':  alpha_corr_rad,
        'alpha_corr_deg':  alpha_corr_deg,
        'corr_rad':        corr,
        'r_min':           r_min_val,
        'b':               b,
        'r0_sim':          r0_val,
        'r_final_sim':     r_f_val,
        'alpha_WF_rad':    alpha_WF_rad,
        'alpha_WF_deg':    alpha_WF_deg,
    }


# ==============================================================================
# Diagnóstico completo (actualizado con mejoras 3 y 6)
# ==============================================================================

def diagnostico_completo(resultado):
    """
    Genera un texto de diagnóstico físico a partir del resultado de la integración.

    Muestra: tipo de trayectoria (clasificación detallada), E, L, b=L/E,
    r_min, r_max, λ_total, advertencia de rdot0 manual si procede,
    y métricas de conservación de energía y precesión del perihelio.
    """
    E = resultado['E']
    L = resultado['L']
    eps = resultado['epsilon']
    tipo = resultado['tipo']
    r_min = resultado.get('r_min', float('nan'))
    r_max = resultado.get('r_max', float('nan'))
    lam_total = resultado['lam'][-1] if len(resultado['lam']) > 0 else 0.0
    b = abs(L / E) if abs(E) > 1e-10 else float('inf')

    tipo_part = "Fotón" if eps == 0 else "Partícula masiva"

    # Usar clasificar_trayectoria_teorica para obtener la descripción detallada:
    # distingue ISCO, circular estable/inestable, esfera de fotones, ligada, etc.
    if tipo == 'error':
        tipo_traj = 'Error de integración'
    else:
        r0_val = float(resultado['r'][0]) if len(resultado['r']) > 0 else None
        tipo_traj = clasificar_trayectoria_teorica(E, L, eps, r0=r0_val)

    lineas = []

    # Mostrar advertencia de rdot0 manual al inicio del diagnóstico si existe
    aviso = resultado.get('aviso_rdot0')
    if aviso:
        lineas += [aviso, "-" * 34]

    lineas += [
        f"Tipo de particula  : {tipo_part}",
        f"Tipo de trayectoria: {tipo_traj}",
        "-" * 34,
        f"Energia E          : {E:.6f}",
        f"Mom. angular L     : {L:.6f}",
        f"Param. impacto b   : {b:.6f}",
        "-" * 34,
        f"r minimo alcanzado : {r_min:.4f} M",
        f"r maximo alcanzado : {r_max:.4f} M",
        f"lambda total       : {lam_total:.2f}",
    ]

    # --- Mejora 6: Tiempo propio (solo para partículas masivas) ---
    tp = calcular_tiempo_propio(resultado)
    if tp is not None:
        lineas += [
            "-" * 34,
            "Dilatacion temporal integrada:",
            f"  Tiempo propio tau : {tp['tau']:.4f} M",
            f"  Tiempo coord. t   : {tp['t_coo']:.4f} M",
            f"  Razon tau/t       : {tp['ratio']:.6f}",
            f"  Atraso t-tau      : {tp['atraso']:.4f} M",
        ]

    # --- Mejora 3: Precesion del perihelio (solo para orbitas ligadas) ---
    prec = calcular_precesion_perihelio(resultado)
    if prec is not None:
        err_prec = abs(prec['exceso_num_rad'] - prec['exceso_anal_rad']) \
                   / max(abs(prec['exceso_anal_rad']), 1e-12) * 100
        # La formula de 1er orden solo es valida para e<<1 (orbitas casi circulares)
        # Para e grande el error es esperado (dominan terminos de orden superior)
        nota_e = " [1er orden valido]" if prec['excentricidad'] < 0.2 else \
                 " [e grande: formula approx.]"
        lineas += [
            "-" * 34,
            f"Precesion perihelio ({prec['n_periapsis']} periapsis):",
            f"  r_peri = {prec['r_peri']:.4f} M",
            f"  r_apo  = {prec['r_apo']:.4f} M",
            f"  e = {prec['excentricidad']:.4f}  p = {prec['semi_latus_rectum']:.4f} M",
            f"  Delta_phi (numerico)  : {prec['dphi_medio_rad']:.6f} rad",
            f"  Exceso numer.         : {prec['exceso_num_rad']:.6f} rad",
            f"  Exceso 6pi/p (1er ord): {prec['exceso_anal_rad']:.6f} rad{nota_e}",
            f"  Discrepancia          : {err_prec:.1f}%",
        ]

    # --- Ángulo de deflexión (solo fotones de dispersión) ---
    defl = calcular_angulo_deflexion(resultado)
    if defl is not None:
        lineas += [
            "-" * 34,
            "Angulo de deflexion gravitacional:",
            f"  b (param. impacto)  : {defl['b']:.4f} M",
            f"  r_min alcanzado     : {defl['r_min']:.4f} M",
            f"  |Δφ| (simulacion)   : {defl['delta_phi_sim']:.6f} rad",
            f"  α = |Δφ| − π       : {defl['alpha_rad']:.6f} rad  ({defl['alpha_deg']:.3f}°)",
            f"  α_corr (→∞)        : {defl['alpha_corr_rad']:.6f} rad  ({defl['alpha_corr_deg']:.3f}°)",
            f"  α_WF = 4/b         : {defl['alpha_WF_rad']:.6f} rad  ({defl['alpha_WF_deg']:.3f}°)",
        ]
    elif eps == 0 and tipo != 'error':
        lineas += [
            "-" * 34,
            "Angulo de deflexion: No aplicable",
            "  (captura, esfera de fotones o sin periastro claro)",
        ]

    # --- Mejora 2: Calidad de la conservacion de energia ---
    _, delta = residuo_conservacion(resultado)
    if len(delta) > 0:
        max_residuo = float(np.max(delta))
        lineas += [
            "-" * 34,
            f"Conservacion energia (DOP853):",
            f"  max|E^2-Veff-rdot^2| = {max_residuo:.2e}",
        ]

    if resultado.get('mensaje'):
        lineas.append(f"Integrador: {resultado['mensaje']}")

    return "\n".join(lineas)
