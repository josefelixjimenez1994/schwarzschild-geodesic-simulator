"""
visualization.py — Funciones de visualización matplotlib para geodésicas de Schwarzschild.

Cada función recibe un Axes de matplotlib y lo rellena con la gráfica correspondiente.
Esto permite usarlas tanto en la interfaz gráfica como para exportación directa.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from metric import f, R_HORIZONTE, R_FOTON, R_ISCO, dilatacion_temporal
from integrator import residuo_conservacion, calcular_precesion_perihelio
from potentials import (
    potencial_efectivo, encontrar_puntos_retorno,
    encontrar_radios_circulares, regiones_permitidas
)

# ==============================================================================
# Paleta de colores
# ==============================================================================

C = {
    'horizonte':   '#FF4444',
    'foton':       '#FF9900',
    'isco':        '#44DD44',
    'trayectoria': '#4488FF',
    'potencial':   '#AA66FF',
    'energia':     '#FF4488',
    'fondo_fig':   '#1A1A2E',
    'fondo_ax':    '#16213E',
    'rejilla':     '#2A2A4A',
    'texto':       '#E0E0E0',
    'punto_ret':   '#FFFFFF',
    'agujero':     '#000000',
    'permitido':   '#1A4A1A',
    'prohibido':   '#4A1A1A',
    'marcador':    '#FFFF00',
    'periastro':   '#00FFFF',
}


def configurar_estilo_oscuro():
    """Aplica estilo oscuro global a todas las figuras matplotlib."""
    plt.rcParams.update({
        'figure.facecolor':  C['fondo_fig'],
        'axes.facecolor':    C['fondo_ax'],
        'axes.edgecolor':    '#4A4A6A',
        'axes.labelcolor':   C['texto'],
        'axes.titlecolor':   C['texto'],
        'xtick.color':       C['texto'],
        'ytick.color':       C['texto'],
        'text.color':        C['texto'],
        'grid.color':        C['rejilla'],
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'legend.facecolor':  '#0D1117',
        'legend.edgecolor':  '#4A4A6A',
        'legend.labelcolor': C['texto'],
        'font.size':         9,
        'axes.titlesize':    10,
        'axes.labelsize':    9,
        'figure.dpi':        100,
    })


# ==============================================================================
# Panel 1: Trayectoria orbital
# ==============================================================================

def dibujar_circulos_referencia(ax, r_lim=None, M_solar=None):
    """
    Dibuja los tres círculos de referencia de Schwarzschild y el agujero negro.

    Si M_solar no es None, añade la distancia en km a las etiquetas.
    """
    def _etiqueta(nombre, r_M, M_sol):
        if M_sol is None:
            return nombre
        from units import r_a_km, _fmt
        km = r_a_km(r_M, M_sol)
        return f"{nombre}  ({_fmt(km)})"

    referencias = [
        (R_HORIZONTE, C['horizonte'],
         _etiqueta('Horizonte r=2M', 2.0, M_solar), '-',  2.0),
        (R_FOTON,     C['foton'],
         _etiqueta('Esf.fotones r=3M', 3.0, M_solar), '--', 1.5),
        (R_ISCO,      C['isco'],
         _etiqueta('ISCO r=6M', 6.0, M_solar), ':',  1.5),
    ]
    for r, color, etiqueta, ls, lw in referencias:
        if r_lim is None or r < r_lim * 1.2:
            c = Circle((0, 0), r, fill=False, color=color,
                       linestyle=ls, linewidth=lw, label=etiqueta, alpha=0.85, zorder=6)
            ax.add_patch(c)

    agujero = Circle((0, 0), R_HORIZONTE, fill=True,
                     color=C['agujero'], alpha=1.0, zorder=7)
    ax.add_patch(agujero)


def graficar_trayectoria(ax, resultado, titulo=None, color=None,
                         etiqueta=None, limpiar=True, M_solar=None):
    """
    Dibuja la trayectoria orbital r(φ) en coordenadas cartesianas.

    Parámetros:
        ax       : Axes de matplotlib
        resultado: dict devuelto por integrar_geodesica()
        titulo   : título del panel (None = automático)
        color    : color de la línea (None = azul por defecto)
        etiqueta : etiqueta para la leyenda
        limpiar  : si True, limpia el axes antes de dibujar
    """
    if limpiar:
        ax.cla()

    r = resultado['r']
    phi = resultado['phi']
    tipo = resultado['tipo']

    x = r * np.cos(phi)
    y_coord = r * np.sin(phi)

    r_max = float(np.max(r))
    lim = min(r_max * 1.15, 80.0)

    if color is None:
        color = C['trayectoria']
    if etiqueta is None:
        etiqueta = f"Trayectoria ({tipo})"

    # Trazar la trayectoria con gradiente de color según el parámetro afín
    ax.plot(x, y_coord, color=color, linewidth=1.6, alpha=0.9,
            label=etiqueta, zorder=10)

    # Punto de inicio
    ax.plot(x[0], y_coord[0], 'o', color='#00FF88', markersize=7,
            label='Inicio', zorder=15)

    # Punto final (captura al horizonte, escape, o final del parámetro afín)
    ax.plot(x[-1], y_coord[-1], 's', color='#FF8800', markersize=7,
            label='Final', zorder=15)

    # Periastros y precesión relativista para órbitas ligadas no circulares
    prec = None
    if tipo == 'ligada':
        prec = calcular_precesion_perihelio(resultado)
        if prec is not None:
            rdot = resultado['rdot']
            primera = True
            for i in range(1, len(rdot)):
                if rdot[i - 1] < 0.0 and rdot[i] >= 0.0:
                    lbl = 'Periastro' if primera else ''
                    ax.plot(x[i], y_coord[i], 'o',
                            color=C['periastro'], markersize=9,
                            label=lbl, zorder=20)
                    primera = False

    dibujar_circulos_referencia(ax, r_lim=lim, M_solar=M_solar)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x / M')
    ax.set_ylabel('y / M')

    if titulo is None:
        eps = resultado.get('epsilon', 1)
        tipo_part = "Fotón" if eps == 0 else "Partícula"
        sufijo_M = "" if M_solar is None else f"  |  M = {M_solar:.3g} M☉"
        titulo = f"Trayectoria orbital — {tipo_part} ({tipo}){sufijo_M}"
    ax.set_title(titulo)

    # Añadir la precesión como entrada adicional en la leyenda
    handles, labels = ax.get_legend_handles_labels()
    if prec is not None:
        exceso = prec['dphi_medio_rad'] - 2.0 * np.pi
        h = Line2D([], [], color='none',
                   label=f'Precesión: Δφ = {exceso:.3f} rad/órbita')
        handles.append(h)
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)


# ==============================================================================
# Panel 2: Potencial efectivo
# ==============================================================================

def graficar_potencial(ax, L, epsilon, E=None, r_min=2.05, r_max=40.0,
                       mostrar_regiones=True):
    """
    Dibuja el potencial efectivo V_eff(r) y, opcionalmente, la línea E².

    Etiqueta los radios críticos y colorea las regiones permitida/prohibida.
    Parámetro E puede ser None para mostrar solo el potencial.
    """
    ax.cla()

    r = np.linspace(r_min, r_max, 5000)
    V = potencial_efectivo(r, L, epsilon)

    # Colorear región permitida/prohibida si E está definida
    if E is not None and mostrar_regiones:
        E2 = E**2
        permitido = V <= E2
        ax.fill_between(r, 0, V,
                        where=permitido, color=C['permitido'],
                        alpha=0.3, label='Región permitida')
        ax.fill_between(r, 0, V,
                        where=~permitido, color=C['prohibido'],
                        alpha=0.3, label='Región prohibida')

    # Curva del potencial
    ax.plot(r, V, color=C['potencial'], linewidth=2.0,
            label=r'$V_{eff}(r)$', zorder=10)

    # Línea de energía E²
    if E is not None:
        E2 = E**2
        ax.axhline(y=E2, color=C['energia'], linewidth=1.8,
                   linestyle='--', label=f'$E^2 = {E2:.4f}$', zorder=9)

        # Puntos de retorno
        puntos = encontrar_puntos_retorno(E, L, epsilon, r_min=r_min + 0.01, r_max=r_max)
        for i, pr in enumerate(puntos):
            ax.axvline(x=pr, color=C['punto_ret'], linewidth=0.8,
                       linestyle=':', alpha=0.7, zorder=8)
            ax.plot(pr, E2, '^', color=C['punto_ret'], markersize=9,
                    zorder=15,
                    label=f'Retorno r={pr:.3f}M' if i == 0 else f'Retorno r={pr:.3f}M')

    # Extremos del potencial (órbitas circulares)
    radios_circ = encontrar_radios_circulares(L, epsilon)
    if radios_circ:
        for rc, tipo_c in zip(radios_circ,
                              ['estable (mín)', 'inestable (máx)']):
            if rc is not None and r_min < rc < r_max:
                Vc = potencial_efectivo(rc, L, epsilon)
                ax.plot(rc, Vc, '*', color='#FFFF00', markersize=14,
                        zorder=16, label=f'Circ. {tipo_c} r={rc:.3f}M')

    # Líneas de radios críticos
    for r_ref, color, etiqueta in [
        (R_HORIZONTE, C['horizonte'], 'r=2M'),
        (R_FOTON,     C['foton'],    'r=3M'),
        (R_ISCO,      C['isco'],     'r=6M'),
    ]:
        if r_min < r_ref < r_max:
            ax.axvline(x=r_ref, color=color, linewidth=1.2,
                       linestyle='--', alpha=0.7, label=etiqueta)

    # Ajustar límites del eje y con márgenes.
    # Si E está definido, garantizar que la línea E² sea siempre visible.
    V_finitos = V[np.isfinite(V)]
    y_max = min(np.percentile(V_finitos, 99) * 1.3, 5.0) if len(V_finitos) else 3.0
    if E is not None:
        y_max = max(y_max, E ** 2 * 1.1)
    y_min = -0.05

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('r / M')
    ax.set_ylabel(r'$V_{eff}(r)$')

    tipo_str = "fotón" if epsilon == 0 else "masiva"
    ax.set_title(f'Potencial efectivo (L={L:.3f}, {tipo_str})')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)


# ==============================================================================
# Panel 3: Evolución radial r(λ)
# ==============================================================================

def _eliminar_twin_axes(ax):
    """
    Elimina los ejes gemelos (twinx) asociados a ax, si existen.
    Necesario para limpiar correctamente entre redibujos.
    """
    fig = ax.get_figure()
    pos_ax = ax.get_position()
    for a in fig.get_axes()[:]:   # copia de la lista para iterar de forma segura
        if a is not ax and np.allclose(
            [a.get_position().x0, a.get_position().y0],
            [pos_ax.x0, pos_ax.y0], atol=0.01
        ):
            fig.delaxes(a)


def graficar_evolucion_radial(ax, resultado, limpiar=True):
    """
    Dibuja r(λ) en el eje principal y, en un eje secundario (derecho),
    el residuo de conservación de energía |E²-V_eff-ṙ²| en escala logarítmica.

    El residuo debe ser ≈ 0 (≤ 1e-9 con DOP853): valida la precisión numérica.
    Los periapsis de órbitas ligadas se marcan con puntos amarillos.
    """
    if limpiar:
        _eliminar_twin_axes(ax)
        ax.cla()

    lam  = resultado['lam']
    r    = resultado['r']
    tipo = resultado['tipo']

    # --- Curva principal r(λ) ---
    ax.plot(lam, r, color=C['trayectoria'], linewidth=1.6,
            label=f'r(λ) — {tipo}', zorder=5)

    # --- Líneas de radios críticos ---
    for r_ref, color, etiqueta in [
        (R_HORIZONTE, C['horizonte'], 'Horizonte (2M)'),
        (R_FOTON,     C['foton'],    'Esf. fotones (3M)'),
        (R_ISCO,      C['isco'],     'ISCO (6M)'),
    ]:
        ax.axhline(y=r_ref, color=color, linewidth=1.2,
                   linestyle='--', alpha=0.75, label=etiqueta, zorder=4)

    # --- Mejora 3: Marcar periapsis solo en órbitas ligadas no circulares ---
    if tipo == 'ligada' and calcular_precesion_perihelio(resultado) is not None:
        rdot = resultado['rdot']
        for i in range(1, len(rdot)):
            if rdot[i - 1] < 0.0 and rdot[i] >= 0.0:
                ax.plot(lam[i], r[i], '^', color=C['periastro'], markersize=6,
                        zorder=10, label='Periapsis' if i == 1 else '')

    ax.set_xlabel('λ (parámetro afín)')
    ax.set_ylabel('r / M')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.4)

    # --- Mejora 2: Eje secundario — residuo de conservación ---
    lam_res, delta = residuo_conservacion(resultado)
    delta = np.maximum(delta, 1e-20)   # evitar log(0)

    ax2 = ax.twinx()
    ax2.semilogy(lam_res, delta, color='#FF8800', linewidth=0.9,
                 linestyle='--', alpha=0.8, label='|E²-Veff-ṙ²|')
    ax2.set_ylabel('Residuo conservación', color='#FF8800', fontsize=7)
    ax2.tick_params(axis='y', colors='#FF8800', labelsize=7)
    ax2.set_facecolor('none')

    # Rango del eje secundario: de 1e-14 a 1e-4 (cubriendo rango DOP853)
    ax2.set_ylim(1e-16, 1e-3)
    ax2.legend(loc='lower right', fontsize=7)

    max_delta = float(np.max(delta))
    ax.set_title(f'r(λ)  |  conserv. max={max_delta:.1e}')


# ==============================================================================
# Panel 4: Dilatación temporal gravitatoria
# ==============================================================================

def graficar_dilatacion_estatica_general(ax, r_min=2.001, r_max=30.0):
    """
    Curva estática dτ/dt = √(1-2M/r) frente a r/M.
    Solo depende de la geometría: válida sin ninguna simulación.
    Usada en la pestaña independiente "Referencia estática".
    Dominio físico: solo exterior de Schwarzschild (r > 2M).
    El eje x empieza en 0 para mostrar el origen, pero la curva y el
    sombreado dejan claro que r < 2M es región no exterior.
    """
    ax.cla()

    # Región r < 2M: sombreada para indicar que no pertenece al exterior
    ax.axvspan(0, R_HORIZONTE, color='#cc3333', alpha=0.10, zorder=1,
               label=r'Región no exterior ($r < 2M$)')

    # Curva solo para r > 2M; nunca incluye el horizonte ni el interior
    r_curva  = np.linspace(max(r_min, 2.001), r_max, 2000)
    fr_curva = 1.0 - 2.0 / r_curva
    dtau_dt  = np.where(fr_curva > 0, np.sqrt(fr_curva), np.nan)

    ax.plot(r_curva, dtau_dt, color=C['potencial'], linewidth=2.0,
            label=r'Ref. estática: $d\tau/dt = \sqrt{1-2M/r}$', zorder=5)

    # Horizonte: siempre visible; la curva se aproxima a 0 desde la derecha
    ax.axvline(x=R_HORIZONTE, color=C['horizonte'], linewidth=1.8,
               linestyle='--', alpha=0.9, label=r'Horizonte $r=2M$', zorder=4)

    for r_ref, color, etiqueta in [
        (R_FOTON, C['foton'], fr'$r=3M$  ($d\tau/dt={np.sqrt(1/3):.3f}$)'),
        (R_ISCO,  C['isco'],  fr'$r=6M$  ($d\tau/dt={np.sqrt(2/3):.3f}$)'),
    ]:
        if r_ref <= r_max:
            ax.axvline(x=r_ref, color=color, linewidth=1.5,
                       linestyle='--', alpha=0.8, label=etiqueta, zorder=4)
            ax.plot(r_ref, dilatacion_temporal(r_ref),
                    'o', color=color, markersize=8, zorder=12)

    ax.axhline(y=1.0, color='#888888', linewidth=0.8,
               linestyle=':', alpha=0.7, label=r'$r \to \infty$:  $d\tau/dt \to 1$')

    ax.set_xlabel('r / M')
    ax.set_ylabel(r'$d\tau/dt$')
    ax.set_title('Referencia estática de dilatación temporal gravitatoria')
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.02, 1.15)
    ax.set_xticks([0, 2, 3, 6, 10, 15, 20, 25, 30])
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.4)


def graficar_dilatacion_temporal(ax, resultado=None, r_min=2.01, r_max=30.0):
    """
    Panel de dilatación temporal del simulador principal. Tres modos:

    - Sin simulación (resultado=None): placeholder con mensaje.
    - Fotón (epsilon=0): mensaje — el fotón no posee tiempo propio.
    - Partícula masiva (epsilon=1): comparación observador estático vs partícula
      a lo largo de la trayectoria frente a τ/M.
    """
    ax.cla()
    # Restaurar ticks completos; el modo fotón los suprime después si procede
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    es_masiva = (resultado is not None
                 and resultado.get('epsilon', 1) == 1
                 and len(resultado['r']) > 1)
    es_foton  = (resultado is not None
                 and resultado.get('epsilon', 1) == 0)

    if resultado is None:
        # ── Sin simulación: placeholder ──────────────────────────────────────
        ax.set_facecolor(C['fondo_ax'])
        ax.set_title('Dilatación temporal de la trayectoria')
        ax.text(0.5, 0.5,
                'Ejecute una simulación\npara mostrar la\ncomparación temporal',
                transform=ax.transAxes, ha='center', va='center',
                color='#666666', fontsize=10)
        ax.set_xlabel(r'$\tau / M$')
        ax.set_ylabel(r'$d\tau/dt$')

    elif es_foton:
        # ── Fotón: sin tiempo propio ─────────────────────────────────────────
        ax.set_facecolor(C['fondo_ax'])
        ax.set_title('Tiempo propio no definido para geodésicas nulas')
        ax.text(0.5, 0.5,
                'Geodésica nula:\nel fotón no posee\ntiempo propio',
                transform=ax.transAxes, ha='center', va='center',
                color='#AAAAFF', fontsize=12, style='italic')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    else:
        # ── Partícula masiva: comparación a lo largo de la trayectoria ───────
        r   = resultado['r']
        lam = resultado['lam']
        E   = resultado['E']

        fr = 1.0 - 2.0 / r
        factor_estatico  = np.where(fr > 0, np.sqrt(fr), np.nan)
        factor_particula = np.where(fr > 0, fr / E,       np.nan)

        ax.plot(lam, factor_estatico,
                color=C['potencial'], linewidth=1.8,
                label='Observador estático en r(τ)', zorder=5)
        ax.plot(lam, factor_particula,
                color=C['trayectoria'], linewidth=1.8,
                label='Partícula simulada', zorder=6)
        ax.axhline(y=1.0, color='#888888', linewidth=0.8,
                   linestyle=':', alpha=0.7, label=r'Referencia $d\tau/dt = 1$')

        validos = np.concatenate([factor_estatico[~np.isnan(factor_estatico)],
                                  factor_particula[~np.isnan(factor_particula)]])
        if len(validos) > 0:
            ymin = max(-0.02, float(np.nanmin(validos)) - 0.05)
            ymax = min(1.15,  float(np.nanmax(validos)) + 0.05)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(-0.02, 1.15)

        ax.set_xlabel(r'$\tau$ / M  (tiempo propio)')
        ax.set_ylabel(r'$d\tau/dt$')
        ax.set_title('Dilatación temporal a lo largo de la trayectoria')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.4)


# ==============================================================================
# Explorador de potencial
# ==============================================================================

def graficar_potencial_explorador(ax, L, epsilon, E, r_min=2.05, r_max=50.0):
    """
    Versión ampliada del gráfico del potencial para el modo explorador.
    Añade anotaciones de texto con los valores de los extremos.
    """
    graficar_potencial(ax, L, epsilon, E, r_min, r_max, mostrar_regiones=True)

    # Guardar ylim fijado por graficar_potencial antes de añadir anotaciones
    ylim = ax.get_ylim()

    # Anotaciones adicionales con valores numéricos.
    # textcoords="offset points" evita que el desplazamiento del texto
    # se exprese en unidades del eje Y, lo que causaría que matplotlib
    # expandiera el eje para incluir el texto y aplastara la curva.
    radios_circ = encontrar_radios_circulares(L, epsilon)
    if radios_circ:
        for rc, nombre in zip(radios_circ, ['Mínimo\n(estable)', 'Máximo\n(inestable)']):
            if rc is not None and r_min < rc < r_max:
                Vc = potencial_efectivo(rc, L, epsilon)
                ax.annotate(
                    f'{nombre}\nr={rc:.2f}M\nVeff={Vc:.4f}',
                    xy=(rc, Vc), xytext=(40, 20),
                    textcoords='offset points',
                    fontsize=7, color='#FFFF00',
                    arrowprops=dict(arrowstyle='->', color='#FFFF00', lw=0.8),
                )

    # Restaurar ylim: las anotaciones no deben modificar la escala del eje Y
    ax.set_ylim(ylim)

    # Mostrar b=|L/E| si E > 0 (el parámetro de impacto se toma siempre positivo)
    if abs(E) > 1e-6:
        b = abs(L / E)
        ax.set_title(
            f'Potencial efectivo  |  L={L:.3f}, E={E:.3f}, b=|L/E|={b:.3f}',
            fontsize=9
        )


# ==============================================================================
# Comparador de trayectorias
# ==============================================================================

def graficar_comparacion(ax, lista_resultados, lista_configs, r_lim=None):
    """
    Dibuja varias trayectorias en el mismo panel.

    Parámetros:
        ax             : Axes de matplotlib
        lista_resultados: lista de dicts devueltos por integrar_geodesica()
        lista_configs  : lista de dicts de configuración con 'etiqueta' y 'color'
        r_lim          : límite radial del gráfico (None = automático)
    """
    ax.cla()

    r_maxs = []
    for resultado, config in zip(lista_resultados, lista_configs):
        r = resultado['r']
        phi = resultado['phi']
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        color = config.get('color', '#4488FF')
        etiqueta = config.get('etiqueta', f"b={config.get('L', '?'):.3f}")

        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9,
                label=etiqueta, zorder=10)
        r_maxs.append(float(np.max(r)))

    lim = min(max(r_maxs) * 1.15, 80.0) if r_maxs else 30.0
    if r_lim is not None:
        lim = r_lim

    dibujar_circulos_referencia(ax, r_lim=lim)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x / M')
    ax.set_ylabel('y / M')
    ax.set_title('Comparador de trayectorias')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.4)


# ==============================================================================
# Exportación de figuras
# ==============================================================================

def exportar_figura(fig, ruta_base, dpi=200):
    """
    Guarda la figura en PNG de alta resolución y PDF vectorial.

    Parámetros:
        fig       : Figure de matplotlib
        ruta_base : ruta sin extensión (e.g. 'figuras/orbita')
        dpi       : resolución para PNG (200 es suficiente para TFG)
    """
    import os
    os.makedirs(os.path.dirname(ruta_base) or '.', exist_ok=True)

    ruta_png = ruta_base + '.png'
    ruta_pdf = ruta_base + '.pdf'

    fig.savefig(ruta_png, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    fig.savefig(ruta_pdf, bbox_inches='tight',
                facecolor=fig.get_facecolor())

    return ruta_png, ruta_pdf
