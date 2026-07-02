"""
gui.py — Interfaz gráfica principal con PyQt6 + Matplotlib.

Estructura de la ventana:
    QTabWidget
    ├── Tab 1: Simulador principal (4 paneles + controles + animación)
    ├── Tab 2: Explorador de potencial (deslizadores en tiempo real)
    └── Tab 3: Comparador de trayectorias (múltiples trayectorias)

Botón "Exportar para TFG" disponible en todas las pestañas.
"""

import sys
import os
import numpy as np

from units import MASAS_PREDEFINIDAS, texto_escala, radios_criticos_km, r_a_km

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QApplication, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QGroupBox, QSlider,
    QTextEdit, QScrollArea, QSizePolicy, QFileDialog,
    QStatusBar, QFrame, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QMessageBox, QDialog, QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from integrator import integrar_geodesica, diagnostico_completo
from potentials import L_circular, E_circular, clasificar_trayectoria_teorica
from visualization import (
    configurar_estilo_oscuro,
    graficar_trayectoria, graficar_potencial, graficar_evolucion_radial,
    graficar_dilatacion_temporal, graficar_dilatacion_estatica_general,
    graficar_potencial_explorador,
    graficar_comparacion,
    C as COLORES,
)
from presets import (
    PRESETS_MASIVAS, PRESETS_FOTONES, trayectorias_comparacion_bc,
    trayectorias_deflexion_foton,
    B_CRITICO,
)
from metric import R_HORIZONTE, R_FOTON, R_ISCO

# Aplicar estilo oscuro a matplotlib al cargar el módulo
configurar_estilo_oscuro()


# ==============================================================================
# Hilo de trabajo para integración en segundo plano (no bloquea la GUI)
# ==============================================================================

class HiloIntegracion(QThread):
    """Ejecuta integrar_geodesica() en un hilo separado."""

    terminado = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            resultado = integrar_geodesica(**self.params)
            self.terminado.emit(resultado)
        except Exception as e:
            self.error.emit(str(e))


# ==============================================================================
# Controlador de animación
# ==============================================================================

class AnimacionControlador:
    """
    Controla la animación de la trayectoria usando un QTimer.
    Mueve marcadores simultáneos en tres paneles:
      - Panel orbital (x, y)
      - Panel r(λ)
      - Panel dilatación temporal dτ/dt  [mejora 4]
    """

    VELOCIDADES = [1, 2, 5, 10, 20, 50]

    def __init__(self, canvas, ax_traj, ax_radial, ax_dilat, resultado):
        self.canvas    = canvas
        self.ax_traj   = ax_traj
        self.ax_radial = ax_radial
        self.ax_dilat  = ax_dilat      # panel dτ/dt para el marcador dinámico
        self.resultado = resultado
        self.timer     = QTimer()
        self.timer.timeout.connect(self._siguiente_frame)

        self.frame         = 0
        self.idx_velocidad = 0
        self.activo        = False

        r   = resultado['r']
        phi = resultado['phi']
        self.x_traj  = r * np.cos(phi)
        self.y_traj  = r * np.sin(phi)
        self.lam     = resultado['lam']
        self.r_traj  = r
        self.n       = len(r)

        # Precalcular factor temporal para la animación del panel dτ/dt (mejora 4)
        from metric import dilatacion_temporal
        eps_anim = resultado.get('epsilon', 1)
        self._animar_masiva = (eps_anim == 1)
        if self._animar_masiva:
            fr_anim = 1.0 - 2.0 / r
            self.dtau_dt_traj = np.where(fr_anim > 0, fr_anim / resultado['E'], 0.0)
        else:
            self.dtau_dt_traj = dilatacion_temporal(r)

        # Marcadores — se crean en iniciar()
        self.marcador_orbital = None
        self.marcador_radial  = None
        self.marcador_dilat   = None   # marcador dinámico sobre curva dτ/dt

    def iniciar(self):
        if self.marcador_orbital is None:
            self.marcador_orbital, = self.ax_traj.plot(
                [], [], 'o', color=COLORES['marcador'],
                markersize=11, zorder=20, label='Particula')
            self.marcador_radial, = self.ax_radial.plot(
                [], [], 'o', color=COLORES['marcador'],
                markersize=9, zorder=20)
            # Mejora 4: marcador sobre dτ/dt solo para partículas masivas
            if self._animar_masiva:
                self.marcador_dilat, = self.ax_dilat.plot(
                    [], [], 'D', color=COLORES['marcador'],
                    markersize=10, zorder=20, label='Posicion actual')
                self.ax_dilat.legend(loc='lower right', fontsize=7)
            self.ax_traj.legend(loc='upper right', fontsize=7, ncol=2)

        self.frame = 0
        self.activo = True
        self.timer.start(50)  # ~20 fps

    def pausar_reanudar(self):
        if self.timer.isActive():
            self.timer.stop()
            self.activo = False
        else:
            self.timer.start(50)
            self.activo = True

    def reiniciar(self):
        self.timer.stop()
        self.activo = False
        self.frame = 0
        for m in [self.marcador_orbital, self.marcador_radial, self.marcador_dilat]:
            if m is not None:
                m.set_data([], [])
        self.canvas.draw_idle()

    def detener(self):
        self.timer.stop()
        self.activo = False

    def set_velocidad(self, idx):
        self.idx_velocidad = max(0, min(idx, len(self.VELOCIDADES) - 1))

    def _siguiente_frame(self):
        paso = self.VELOCIDADES[self.idx_velocidad]
        self.frame = min(self.frame + paso, self.n - 1)

        xi  = self.x_traj[self.frame]
        yi  = self.y_traj[self.frame]
        ri  = self.r_traj[self.frame]
        li  = self.lam[self.frame]
        dti = self.dtau_dt_traj[self.frame]

        if self.marcador_orbital:
            self.marcador_orbital.set_data([xi], [yi])
        if self.marcador_radial:
            self.marcador_radial.set_data([li], [ri])
        # Mejora 4: actualizar marcador sobre el panel dτ/dt
        if self.marcador_dilat:
            xi_dilat = li if self._animar_masiva else ri
            self.marcador_dilat.set_data([xi_dilat], [dti])

        self.canvas.draw_idle()

        if self.frame >= self.n - 1:
            self.timer.stop()
            self.activo = False


# ==============================================================================
# Tab 1: Simulador principal
# ==============================================================================

# ==============================================================================
# Diálogo de diagnóstico completo
# ==============================================================================

class DialogDiagnostico(QDialog):
    """Ventana emergente con el texto completo del diagnóstico."""

    def __init__(self, texto, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnóstico completo")
        self.resize(620, 440)
        layout = QVBoxLayout(self)

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setFont(QFont("Consolas", 9))
        editor.setStyleSheet(
            "background-color: #0D1117; color: #88FF88; border: 1px solid #333;"
        )
        editor.setPlainText(texto)
        layout.addWidget(editor)

        btn_cerrar = QPushButton("Cerrar")
        btn_cerrar.clicked.connect(self.accept)
        layout.addWidget(btn_cerrar)


# ==============================================================================
# Widget de selección de masa del agujero negro
# ==============================================================================

class GrupoMasa(QGroupBox):
    """
    Grupo de controles para elegir la masa del agujero negro.
    La simulación corre siempre en M=1; este widget traduce los
    resultados a unidades físicas (km) para mostrarlos en pantalla.
    """

    masa_cambiada = pyqtSignal(float)   # emite M en masas solares

    def __init__(self, parent=None):
        super().__init__("Masa del agujero negro", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Selector de masa predefinida
        self.cmb_masa = QComboBox()
        for nombre in MASAS_PREDEFINIDAS:
            self.cmb_masa.addItem(nombre)
        self.cmb_masa.setCurrentIndex(0)
        layout.addWidget(self.cmb_masa)

        # Spin para valor personalizado
        fila = QHBoxLayout()
        fila.addWidget(QLabel("M / M☉ ="))
        self.spin_masa = QDoubleSpinBox()
        self.spin_masa.setRange(0.1, 1e12)
        self.spin_masa.setDecimals(3)
        self.spin_masa.setValue(10.0)
        self.spin_masa.setSingleStep(1.0)
        self.spin_masa.setStepType(
            QDoubleSpinBox.StepType.AdaptiveDecimalStepType
        )
        fila.addWidget(self.spin_masa)
        layout.addLayout(fila)

        # Panel de radios en km
        self.txt_radios = QTextEdit()
        self.txt_radios.setReadOnly(True)
        self.txt_radios.setMaximumHeight(115)
        self.txt_radios.setFont(QFont("Consolas", 8))
        self.txt_radios.setStyleSheet(
            "background-color: #0D1117; color: #AAFFAA; border: 1px solid #333;"
        )
        layout.addWidget(self.txt_radios)

        # Conexiones
        self.cmb_masa.currentIndexChanged.connect(self._preset_cambiado)
        self.spin_masa.valueChanged.connect(self._spin_cambiado)

        self._actualizar_display(10.0)

    def _preset_cambiado(self, idx):
        nombre = self.cmb_masa.currentText()
        M = MASAS_PREDEFINIDAS[nombre]
        self.spin_masa.blockSignals(True)
        self.spin_masa.setValue(M)
        self.spin_masa.blockSignals(False)
        self._actualizar_display(M)
        self.masa_cambiada.emit(M)

    def _spin_cambiado(self, M):
        self._actualizar_display(M)
        self.masa_cambiada.emit(M)

    def _actualizar_display(self, M):
        self.txt_radios.setPlainText(texto_escala(M))

    def get_masa_solar(self):
        return self.spin_masa.value()


# ==============================================================================
# Panel de controles principal
# ==============================================================================

class PanelControles(QScrollArea):
    """Panel izquierdo con todos los controles de parámetros."""

    calcular_pedido = pyqtSignal(dict)
    csv_pedido      = pyqtSignal()     # mejora 5: solicitud de exportar CSV

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(340)

        contenedor = QWidget()
        self.setWidget(contenedor)
        layout = QVBoxLayout(contenedor)
        layout.setSpacing(6)

        # --- Grupo: Tipo de geodésica ---
        grp_tipo = QGroupBox("Tipo de partícula")
        lay_tipo = QHBoxLayout(grp_tipo)
        self.rb_masiva = QRadioButton("Masiva (ε=1)")
        self.rb_foton  = QRadioButton("Fotón (ε=0)")
        self.rb_masiva.setChecked(True)
        grp_tipo_bg = QButtonGroup(self)
        grp_tipo_bg.addButton(self.rb_masiva)
        grp_tipo_bg.addButton(self.rb_foton)
        lay_tipo.addWidget(self.rb_masiva)
        lay_tipo.addWidget(self.rb_foton)
        layout.addWidget(grp_tipo)

        # --- Grupo: Masa del agujero negro ---
        self.grupo_masa = GrupoMasa()
        layout.addWidget(self.grupo_masa)

        # --- Grupo: Parámetros físicos ---
        grp_params = QGroupBox("Parámetros físicos")
        lay_params = QGridLayout(grp_params)

        self.spin_E    = self._make_spin(0.0, 5.0, 0.9428, 0.001, "Energía E")
        self.spin_L    = self._make_spin(-20.0, 20.0, 3.464, 0.01, "Mom. angular L")
        self.spin_r0   = self._make_spin(2.01, 500.0, 6.0, 0.1, "Radio inicial r₀/M")
        self.spin_phi0 = self._make_spin(-np.pi, np.pi, 0.0, 0.01, "Ángulo φ₀ (rad)")

        etiquetas = ["E", "L", "r₀ / M", "φ₀ (rad)"]
        spins = [self.spin_E, self.spin_L, self.spin_r0, self.spin_phi0]
        for i, (et, sp) in enumerate(zip(etiquetas, spins)):
            lay_params.addWidget(QLabel(et), i, 0)
            lay_params.addWidget(sp, i, 1)

        # Velocidad radial inicial
        self.chk_auto_rdot = QCheckBox("ṙ₀ automático")
        self.chk_auto_rdot.setChecked(True)
        self.spin_rdot0 = self._make_spin(-5.0, 5.0, 0.0, 0.001, "Vel. radial ṙ₀")
        self.spin_rdot0.setEnabled(False)
        self.chk_auto_rdot.toggled.connect(
            lambda checked: self.spin_rdot0.setEnabled(not checked)
        )

        # Sentido de caída (solo activo con ṙ₀ automático)
        self.cmb_sentido = QComboBox()
        self.cmb_sentido.addItems(["Hacia adentro (-1)", "Hacia afuera (+1)"])

        lay_params.addWidget(self.chk_auto_rdot, 4, 0, 1, 2)
        lay_params.addWidget(QLabel("ṙ₀"), 5, 0)
        lay_params.addWidget(self.spin_rdot0, 5, 1)
        lay_params.addWidget(QLabel("Sentido"), 6, 0)
        lay_params.addWidget(self.cmb_sentido, 6, 1)

        layout.addWidget(grp_params)

        # --- Grupo: Parámetros numéricos ---
        grp_num = QGroupBox("Parámetros numéricos")
        lay_num = QGridLayout(grp_num)

        self.spin_lam_max = self._make_spin(10.0, 50000.0, 3000.0, 100.0,
                                            "λ máximo")
        self.spin_lam_max.setDecimals(0)

        self.cmb_tol = QComboBox()
        self.cmb_tol.addItems(["1e-8", "1e-10", "1e-12"])
        self.cmb_tol.setCurrentIndex(1)

        self.cmb_metodo = QComboBox()
        self.cmb_metodo.addItems(["DOP853", "RK45"])

        lay_num.addWidget(QLabel("λ_max"), 0, 0)
        lay_num.addWidget(self.spin_lam_max, 0, 1)
        lay_num.addWidget(QLabel("Tolerancia"), 1, 0)
        lay_num.addWidget(self.cmb_tol, 1, 1)
        lay_num.addWidget(QLabel("Método"), 2, 0)
        lay_num.addWidget(self.cmb_metodo, 2, 1)

        layout.addWidget(grp_num)

        # --- Botón calcular ---
        self.btn_calcular = QPushButton("▶ Calcular geodésica")
        self.btn_calcular.setMinimumHeight(40)
        self.btn_calcular.setStyleSheet(
            "QPushButton { background-color: #2244AA; color: white; "
            "font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #3355CC; }"
            "QPushButton:pressed { background-color: #1133AA; }"
            "QPushButton:disabled { background-color: #555555; }"
        )
        self.btn_calcular.clicked.connect(self._emitir_calcular)
        layout.addWidget(self.btn_calcular)

        # --- Panel de diagnóstico ---
        grp_diag = QGroupBox("Diagnóstico")
        lay_diag = QVBoxLayout(grp_diag)
        self.txt_diagnostico = QTextEdit()
        self.txt_diagnostico.setReadOnly(True)
        self.txt_diagnostico.setMaximumHeight(180)
        self.txt_diagnostico.setFont(QFont("Consolas", 8))
        self.txt_diagnostico.setStyleSheet(
            "background-color: #0D1117; color: #88FF88; border: 1px solid #333;"
        )
        lay_diag.addWidget(self.txt_diagnostico)

        # Botón exportar CSV (mejora 5) — habilitado solo tras calcular
        self.btn_csv = QPushButton("📄 Exportar CSV")
        self.btn_csv.setEnabled(False)
        self.btn_csv.setStyleSheet(
            "QPushButton { background-color: #1A3A2A; color: white; "
            "border-radius: 4px; padding: 3px 8px; }"
            "QPushButton:hover { background-color: #2A5A3A; }"
            "QPushButton:disabled { background-color: #222233; color: #666; }"
        )
        lay_diag.addWidget(self.btn_csv)
        layout.addWidget(grp_diag)

        # Botón para ver el diagnóstico completo en ventana emergente
        self.btn_ver_diag = QPushButton("Ver diagnóstico completo")
        self.btn_ver_diag.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 2px 6px; }"
        )
        self.btn_ver_diag.clicked.connect(self._abrir_diagnostico)
        layout.addWidget(self.btn_ver_diag)

        # --- Controles de animación ---
        grp_anim = QGroupBox("Animación")
        lay_anim = QVBoxLayout(grp_anim)

        # Texto de guía visible antes de la primera simulación
        self.lbl_anim_estado = QLabel("Ejecute una simulación\npara activar la animación")
        self.lbl_anim_estado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_anim_estado.setStyleSheet(
            "color: #777788; font-style: italic; font-size: 9px; padding: 2px 0;"
        )
        lay_anim.addWidget(self.lbl_anim_estado)

        row_btns = QHBoxLayout()
        self.btn_play     = QPushButton("▶ Play")
        self.btn_pausa    = QPushButton("⏸ Pausa")
        self.btn_reinicio = QPushButton("⏹ Reset")
        for btn in [self.btn_play, self.btn_pausa, self.btn_reinicio]:
            btn.setEnabled(False)
            row_btns.addWidget(btn)
        lay_anim.addLayout(row_btns)

        lay_vel = QHBoxLayout()
        lay_vel.addWidget(QLabel("Velocidad:"))
        self.slider_vel = QSlider(Qt.Orientation.Horizontal)
        self.slider_vel.setMinimum(0)
        self.slider_vel.setMaximum(5)
        self.slider_vel.setValue(0)
        self.slider_vel.setEnabled(False)
        self.slider_vel.setTickPosition(QSlider.TickPosition.TicksBelow)
        lay_vel.addWidget(self.slider_vel)
        self.lbl_vel = QLabel("×1")
        lay_vel.addWidget(self.lbl_vel)
        lay_anim.addLayout(lay_vel)

        layout.addWidget(grp_anim)

        # --- Presets ---
        grp_pre = QGroupBox("Presets rápidos")
        lay_pre = QVBoxLayout(grp_pre)
        lay_pre.addWidget(QLabel("Partículas masivas:"))
        for nombre in PRESETS_MASIVAS:
            btn = QPushButton(nombre)
            btn.setStyleSheet("text-align: left; padding: 3px 6px;")
            btn.clicked.connect(lambda checked, n=nombre: self._cargar_preset_masiva(n))
            lay_pre.addWidget(btn)

        lay_pre.addWidget(QLabel("Fotones:"))
        for nombre in PRESETS_FOTONES:
            btn = QPushButton(nombre)
            btn.setStyleSheet("text-align: left; padding: 3px 6px;")
            btn.clicked.connect(lambda checked, n=nombre: self._cargar_preset_foton(n))
            lay_pre.addWidget(btn)

        layout.addWidget(grp_pre)
        layout.addStretch()

        # Almacenamiento del animador actual
        self.animador = None
        # Nombre del último preset cargado (None = parámetros manuales)
        self._preset_activo = None

    # ---- Helpers ----

    def _make_spin(self, minv, maxv, val, step, tooltip=""):
        sp = QDoubleSpinBox()
        sp.setRange(minv, maxv)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setDecimals(6)
        sp.setToolTip(tooltip)
        return sp

    def _emitir_calcular(self):
        params = self._leer_parametros()
        if params:
            self.calcular_pedido.emit(params)

    def _leer_parametros(self):
        epsilon = 0 if self.rb_foton.isChecked() else 1
        sentido = 1 if self.cmb_sentido.currentIndex() == 1 else -1
        tol_str = self.cmb_tol.currentText()
        tol = float(tol_str)
        rdot0 = None if self.chk_auto_rdot.isChecked() else self.spin_rdot0.value()

        return {
            "E":         self.spin_E.value(),
            "L":         self.spin_L.value(),
            "epsilon":   epsilon,
            "r0":        self.spin_r0.value(),
            "phi0":      self.spin_phi0.value(),
            "rdot0":     rdot0,
            "sentido":   sentido,
            "lambda_max": self.spin_lam_max.value(),
            "tolerancia": tol,
            "metodo":    self.cmb_metodo.currentText(),
        }

    def _cargar_preset_masiva(self, nombre):
        self._preset_activo = nombre
        preset = PRESETS_MASIVAS[nombre]
        self._aplicar_preset(preset, masiva=True)

    def _cargar_preset_foton(self, nombre):
        self._preset_activo = nombre
        preset = PRESETS_FOTONES[nombre]
        self._aplicar_preset(preset, masiva=False)

    def _aplicar_preset(self, preset, masiva):
        self.rb_masiva.setChecked(masiva)
        self.rb_foton.setChecked(not masiva)
        self.spin_E.setValue(float(preset['E']))
        self.spin_L.setValue(float(preset['L']))
        self.spin_r0.setValue(float(preset['r0']))
        self.spin_phi0.setValue(float(preset['phi0']))
        self.spin_lam_max.setValue(float(preset['lambda_max']))

        if preset.get('rdot0') is not None:
            self.chk_auto_rdot.setChecked(False)
            self.spin_rdot0.setValue(float(preset['rdot0']))
        else:
            self.chk_auto_rdot.setChecked(True)

        sentido = preset.get('sentido', -1)
        self.cmb_sentido.setCurrentIndex(0 if sentido == -1 else 1)

        # Auto-calcular
        self._emitir_calcular()

    def _abrir_diagnostico(self):
        texto = self.txt_diagnostico.toPlainText()
        dlg = DialogDiagnostico(texto, self)
        dlg.exec()

    def mostrar_diagnostico(self, texto):
        self.txt_diagnostico.setPlainText(texto)

    def habilitar_animacion(self, habilitar):
        for btn in [self.btn_play, self.btn_pausa, self.btn_reinicio]:
            btn.setEnabled(habilitar)
        self.slider_vel.setEnabled(habilitar)
        self.btn_csv.setEnabled(habilitar)
        if habilitar:
            self.lbl_anim_estado.setText("Simulación cargada — listo para animar")
            self.lbl_anim_estado.setStyleSheet(
                "color: #44BB88; font-style: normal; font-size: 9px; padding: 2px 0;"
            )
        else:
            self.lbl_anim_estado.setText("Ejecute una simulación\npara activar la animación")
            self.lbl_anim_estado.setStyleSheet(
                "color: #777788; font-style: italic; font-size: 9px; padding: 2px 0;"
            )

    def get_E(self): return self.spin_E.value()
    def get_L(self): return self.spin_L.value()
    def get_preset_activo(self): return self._preset_activo


class PanelVisualizacion(QWidget):
    """Panel derecho con cuatro subgráficas matplotlib (2×2)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Figura principal con 4 subplots en 2×2
        self.figura = Figure(figsize=(12, 9), tight_layout=True)
        self.figura.patch.set_facecolor(COLORES['fondo_fig'])

        self.ax_traj   = self.figura.add_subplot(2, 2, 1)
        self.ax_pot    = self.figura.add_subplot(2, 2, 2)
        self.ax_radial = self.figura.add_subplot(2, 2, 3)
        self.ax_dilat  = self.figura.add_subplot(2, 2, 4)

        self.canvas = FigureCanvas(self.figura)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Inicializar con gráficas vacías
        self._inicializar_graficas()

        # Datos del último resultado (para ventana ampliada por doble clic)
        self._ultimo_resultado = None
        self._ultimo_M_solar   = None
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if not event.dblclick or event.inaxes is None:
            return
        if self._ultimo_resultado is None:
            return
        mapa = {
            self.ax_traj:   0,
            self.ax_pot:    1,
            self.ax_radial: 2,
            self.ax_dilat:  3,
        }
        panel_idx = mapa.get(event.inaxes)
        if panel_idx is None:
            return
        ventana = VentanaAmpliadaPanel(
            panel_idx, self._ultimo_resultado, self._ultimo_M_solar,
            parent=self.window()
        )
        ventana.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        ventana.show()

    def _inicializar_graficas(self):
        for ax, titulo in [
            (self.ax_traj,   'Trayectoria orbital'),
            (self.ax_pot,    'Potencial efectivo'),
            (self.ax_radial, 'Evolución radial r(λ)'),
            (self.ax_dilat,  'Dilatación temporal'),
        ]:
            ax.set_facecolor(COLORES['fondo_ax'])
            ax.set_title(titulo, color=COLORES['texto'])
            ax.text(0.5, 0.5, 'Ejecute una\nsimulación',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#666666', fontsize=10)

        graficar_dilatacion_temporal(self.ax_dilat)
        self.canvas.draw()

    def actualizar_todo(self, resultado, M_solar=None):
        """Actualiza los 4 paneles con el resultado de la integración."""
        self._ultimo_resultado = resultado
        self._ultimo_M_solar   = M_solar
        E = resultado['E']
        L = resultado['L']
        eps = resultado['epsilon']

        graficar_trayectoria(self.ax_traj, resultado, M_solar=M_solar)
        graficar_potencial(self.ax_pot, L, eps, E)
        graficar_evolucion_radial(self.ax_radial, resultado)
        graficar_dilatacion_temporal(self.ax_dilat, resultado)

        self.canvas.draw_idle()

    def actualizar_potencial(self, L, epsilon, E):
        """Actualiza solo el panel del potencial (al cambiar E o L)."""
        graficar_potencial(self.ax_pot, L, epsilon, E)
        self.canvas.draw_idle()


class TabSimulador(QWidget):
    """Pestaña principal: simulador con 4 paneles y controles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Splitter horizontal: controles | visualización
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.controles = PanelControles()
        self.viz = PanelVisualizacion()

        splitter.addWidget(self.controles)
        splitter.addWidget(self.viz)
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Estado interno
        self.resultado_actual = None
        self.animador = None
        self.hilo = None

        # Conexiones
        self.controles.calcular_pedido.connect(self._iniciar_calculo)
        self.controles.btn_play.clicked.connect(self._play)
        self.controles.btn_pausa.clicked.connect(self._pausa)
        self.controles.btn_reinicio.clicked.connect(self._reiniciar)
        self.controles.slider_vel.valueChanged.connect(self._cambiar_velocidad)

        # Actualizar potencial en tiempo real al cambiar E o L
        self.controles.spin_E.valueChanged.connect(self._actualizar_potencial_live)
        self.controles.spin_L.valueChanged.connect(self._actualizar_potencial_live)
        self.controles.rb_masiva.toggled.connect(self._actualizar_potencial_live)

        # Redibujar la trayectoria cuando cambia la masa (para actualizar etiquetas km)
        self.controles.grupo_masa.masa_cambiada.connect(self._masa_cambiada)

        # Mejora 5: exportar CSV
        self.controles.btn_csv.clicked.connect(self._exportar_csv)

    def _epsilon_actual(self):
        return 0 if self.controles.rb_foton.isChecked() else 1

    def _actualizar_potencial_live(self):
        """Actualiza el panel del potencial cuando el usuario mueve E o L."""
        L   = self.controles.get_L()
        E   = self.controles.get_E()
        eps = self._epsilon_actual()
        self.viz.actualizar_potencial(L, eps, E)

    def _iniciar_calculo(self, params):
        """Lanza la integración en un hilo separado."""
        if self.hilo and self.hilo.isRunning():
            return

        self.controles.btn_calcular.setEnabled(False)
        self.controles.btn_calcular.setText("⏳ Calculando...")
        self.controles.habilitar_animacion(False)

        if self.animador:
            self.animador.detener()
            self.animador = None

        self.hilo = HiloIntegracion(params)
        self.hilo.terminado.connect(self._al_terminar)
        self.hilo.error.connect(self._al_error)
        self.hilo.start()

    def _masa_cambiada(self, M_solar):
        """Redibuja la trayectoria actual con las nuevas etiquetas de km."""
        if self.resultado_actual is not None:
            self.viz.actualizar_todo(self.resultado_actual, M_solar=M_solar)

    @pyqtSlot(dict)
    def _al_terminar(self, resultado):
        self.resultado_actual = resultado
        M_solar = self.controles.grupo_masa.get_masa_solar()
        self.viz.actualizar_todo(resultado, M_solar=M_solar)
        texto = diagnostico_completo(resultado)
        self.controles.mostrar_diagnostico(texto)
        self.controles.btn_calcular.setEnabled(True)
        self.controles.btn_calcular.setText("▶ Calcular geodésica")
        if resultado['exito'] and len(resultado['r']) > 5:
            self.controles.habilitar_animacion(True)

    @pyqtSlot(str)
    def _al_error(self, mensaje):
        self.controles.mostrar_diagnostico(f"ERROR:\n{mensaje}")
        self.controles.btn_calcular.setEnabled(True)
        self.controles.btn_calcular.setText("▶ Calcular geodésica")

    def _play(self):
        if self.resultado_actual is None:
            return
        if self.animador is None:
            # Pasar los 4 ejes: orbital, radial y dilatación temporal (mejora 4)
            self.animador = AnimacionControlador(
                self.viz.canvas,
                self.viz.ax_traj,
                self.viz.ax_radial,
                self.viz.ax_dilat,
                self.resultado_actual,
            )
        self.animador.iniciar()

    def _pausa(self):
        if self.animador:
            self.animador.pausar_reanudar()

    def _reiniciar(self):
        if self.animador:
            self.animador.reiniciar()

    def _cambiar_velocidad(self, valor):
        velocidades = AnimacionControlador.VELOCIDADES
        self.controles.lbl_vel.setText(f"×{velocidades[valor]}")
        if self.animador:
            self.animador.set_velocidad(valor)

    def _exportar_csv(self):
        """
        Mejora 5: Exporta los datos numéricos de la trayectoria actual a CSV.

        Columnas: lambda, t, r, phi, rdot, V_eff(r), E^2-V_eff-rdot^2
        """
        if self.resultado_actual is None:
            return

        res = self.resultado_actual
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Exportar trayectoria CSV",
            os.path.join(os.path.expanduser("~"), "trayectoria_schwarzschild.csv"),
            "CSV (*.csv)"
        )
        if not ruta:
            return

        import csv
        from potentials import potencial_efectivo
        from integrator import residuo_conservacion

        lam  = res['lam']
        t    = res['t']
        r    = res['r']
        phi  = res['phi']
        rdot = res['rdot']
        E    = res['E']
        L    = res['L']
        eps  = res['epsilon']

        V    = potencial_efectivo(r, L, eps)
        _, delta = residuo_conservacion(res)

        from datetime import datetime
        metodo    = self.controles.cmb_metodo.currentText()
        tol_str   = self.controles.cmb_tol.currentText()
        lam_max   = self.controles.spin_lam_max.value()
        tipo_part = "Fotón" if eps == 0 else "Partícula masiva"
        r0_val    = float(r[0])
        fecha     = datetime.now().strftime("%Y-%m-%d %H:%M")

        try:
            with open(ruta, 'w', newline='', encoding='utf-8') as f:
                f.write(
                    "# ====================================================\n"
                    "# SIMULADOR DE GEODÉSICAS DE SCHWARZSCHILD\n"
                    "# Versión: 1.0\n"
                    f"# Fecha: {fecha}\n"
                    f"# Método: {metodo}\n"
                    f"# Tolerancia: {tol_str}\n"
                    f"# λ_max: {lam_max:.0f}\n"
                    "# r_max: 300\n"
                    f"# Tipo: {tipo_part}\n"
                    f"# E: {E:.6f}\n"
                    f"# L: {L:.6f}\n"
                    f"# r0: {r0_val:.4f}\n"
                    "# ====================================================\n"
                )
                escritor = csv.writer(f)
                escritor.writerow(['lambda', 't', 'r', 'phi', 'rdot', 'V_eff', 'E2_minus_Veff_minus_rdot2'])
                for i in range(len(lam)):
                    escritor.writerow([
                        f"{lam[i]:.8e}", f"{t[i]:.8e}", f"{r[i]:.8e}",
                        f"{phi[i]:.8e}", f"{rdot[i]:.8e}",
                        f"{V[i]:.8e}",  f"{delta[i]:.8e}",
                    ])
            QMessageBox.information(self, "CSV exportado", f"Guardado en:\n{ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    def get_figura(self):
        return self.viz.figura

    def get_resultado(self):
        """Devuelve el resultado de la última integración, o None si no hay ninguna."""
        return self.resultado_actual


# ==============================================================================
# Tab 2: Explorador de potencial — lógica de estado físico
# ==============================================================================

def _calcular_estado_fisico(E, L, eps, radios, puntos):
    """
    Clasifica el estado físico esperado sin integrar, a partir de los parámetros
    y la estructura del potencial (radios circulares y puntos de retorno).

    Retorna (emoji, titulo, explicacion_corta, interpretacion_larga).
    """
    E2 = E ** 2

    # ── Fotón ──────────────────────────────────────────────────────────────────
    if eps == 0:
        if abs(E) < 1e-10:
            return ("⬜", "INDEFINIDO",
                    "E = 0 no define parámetro de impacto.",
                    "Parámetros no físicos para un fotón.")
        if abs(L) < 1e-12:
            return (
                "🟥", "CAPTURA RADIAL",
                "El fotón tiene momento angular nulo (L=0) y cae directamente hacia el horizonte.",
                "CAPTURA RADIAL: El fotón tiene momento angular nulo (L=0) y cae directamente "
                "hacia el horizonte. No existe barrera potencial ni parámetro de impacto asociado.",
            )
        b = abs(L / E)
        delta_rel = (b - B_CRITICO) / B_CRITICO
        if abs(delta_rel) < 0.01:
            return (
                "🟨", "ÓRBITA CRÍTICA",
                "El fotón se aproxima asintóticamente a la esfera de fotones r=3M.",
                f"Con b≈b_c ({b:.3f}≈{B_CRITICO:.3f}), el fotón orbita "
                f"asintóticamente en torno a la esfera de fotones r=3M. "
                f"Esta órbita es inestable: cualquier perturbación decide entre "
                f"captura y escape.",
            )
        elif b < B_CRITICO:
            return (
                "🟥", "CAPTURA",
                "El fotón pasa demasiado cerca y no encuentra punto de retorno exterior.",
                f"El potencial presenta una barrera crítica asociada a r≈3M. "
                f"Como b={b:.3f} < b_c={B_CRITICO:.3f}, no existe un punto de retorno "
                f"exterior capaz de invertir el movimiento radial, y el fotón acaba "
                f"siendo capturado por el agujero negro.",
            )
        else:
            return (
                "🟩", "DISPERSIÓN",
                "El fotón es desviado gravitatoriamente y escapa al infinito.",
                f"Con b={b:.3f} > b_c={B_CRITICO:.3f}, el nivel E² queda por debajo "
                f"del máximo del potencial efectivo. El fotón incidente desde el infinito "
                f"alcanza un punto de retorno exterior, es desviado gravitatoriamente "
                f"y vuelve a escapar al infinito.",
            )

    # ── Partícula masiva ───────────────────────────────────────────────────────
    L_ISCO = 2.0 * np.sqrt(3.0)
    E_ISCO = np.sqrt(8.0 / 9.0)

    if (abs(L - L_ISCO) / max(L_ISCO, 1e-10) < 0.02 and
            abs(E - E_ISCO) / E_ISCO < 0.02):
        return (
            "🟨", "ISCO",
            "Última órbita circular estable (r = 6M).",
            "La partícula está en la ISCO (r=6M), la última órbita circular "
            "estable de Schwarzschild. Para r<6M no existen órbitas circulares "
            "estables; cualquier perturbación interior provoca la captura.",
        )

    if radios:
        r_est, r_inest = radios
        if r_est is not None:
            Ec = E_circular(r_est)
            if Ec is not None and abs(E2 - Ec ** 2) < 1e-4 * max(E2, 1e-10):
                return (
                    "🟨", "ÓRBITA CIRCULAR ESTABLE",
                    f"Pequeñas perturbaciones no destruyen la órbita en r={r_est:.2f}M.",
                    f"La energía coincide con el mínimo del potencial en "
                    f"r≈{r_est:.2f}M. La partícula describe una órbita circular "
                    f"estable; perturbaciones pequeñas producen oscilaciones "
                    f"radiales en torno a ese radio.",
                )
        if r_inest is not None:
            Ec = E_circular(r_inest)
            if Ec is not None and abs(E2 - Ec ** 2) < 1e-4 * max(E2, 1e-10):
                return (
                    "🟧", "ÓRBITA CIRCULAR INESTABLE",
                    f"Pequeñas perturbaciones provocan captura o escape desde r={r_inest:.2f}M.",
                    f"La energía coincide con el máximo local del potencial en "
                    f"r≈{r_inest:.2f}M. Esta órbita es inestable: cualquier "
                    f"perturbación hace que la partícula caiga hacia el horizonte "
                    f"o escape al infinito.",
                )

    if len(puntos) == 0:
        if E2 >= 1.0:
            return (
                "🟩", "ESCAPE",
                "La energía supera el umbral de escape (E ≥ 1).",
                "La energía supera el potencial en todo el dominio exterior. "
                "La partícula puede escapar al infinito desde cualquier radio inicial.",
            )
        else:
            return (
                "🟥", "CAPTURA",
                "Sin barrera efectiva: la partícula cae hacia el agujero negro.",
                "Con L insuficiente para crear barrera centrífuga efectiva, "
                "la partícula cae inevitablemente hacia el horizonte de eventos.",
            )

    if len(puntos) == 1:
        pt = puntos[0]
        if E2 >= 1.0:
            return (
                "🟩", "ESCAPE",
                f"Punto de retorno en r≈{pt:.2f}M; la partícula rebota y escapa.",
                f"La partícula tiene un punto de retorno en r≈{pt:.2f}M. "
                f"Con E≥1, la energía es suficiente para escapar al infinito.",
            )
        else:
            return (
                "🟥", "CAPTURA",
                f"Punto de retorno en r≈{pt:.2f}M; sin energía suficiente para escapar.",
                f"La partícula alcanza un mínimo de acercamiento en r≈{pt:.2f}M "
                f"pero no puede escapar al infinito. Termina siendo capturada.",
            )

    if eps == 1 and E2 >= 1.0 and len(puntos) >= 2:
        r_in, r_out = puntos[0], puntos[-1]
        return (
            "🟧", "BARRERA POTENCIAL",
            f"Dos puntos de retorno delimitan una barrera entre r≈{r_in:.2f}M y r≈{r_out:.2f}M.",
            f"Como E≥1, la región exterior se extiende hasta el infinito. "
            f"Una partícula situada en la rama exterior r>r≈{r_out:.2f}M puede escapar "
            f"o sufrir dispersión gravitatoria, mientras que una partícula situada en la "
            f"rama interior 2M<r<r≈{r_in:.2f}M queda asociada a captura. "
            f"Sin especificar r₀ y el sentido radial inicial, el explorador solo identifica "
            f"la estructura de la barrera potencial.",
        )

    # Dos o más puntos de retorno con E² < 1 → órbita ligada
    if eps == 1 and E2 < 1.0 and len(puntos) >= 3:
        r_min_r, r_max_r = puntos[-2], puntos[-1]
    else:
        r_min_r, r_max_r = puntos[0], puntos[-1]

    r_med = (r_min_r + r_max_r) / 2.0
    if r_med > 1e-10 and (r_max_r - r_min_r) / r_med < 0.01:
        return (
            "🟨", "ÓRBITA CIRCULAR ESTABLE",
            f"Órbita casi circular en r≈{r_med:.2f}M.",
            f"Los dos puntos de retorno están muy próximos (r≈{r_med:.2f}M), "
            f"indicando una órbita casi circular estable.",
        )

    return (
        "🟪", "ÓRBITA PRECESANTE",
        f"Órbita ligada entre r≈{r_min_r:.2f}M y r≈{r_max_r:.2f}M.",
        f"La energía corta al potencial en dos puntos de retorno: "
        f"r≈{r_min_r:.2f}M (periastro) y r≈{r_max_r:.2f}M (apoastro). "
        f"La partícula oscila entre ellos describiendo una órbita elíptica "
        f"relativista con precesión del periastro en cada revolución.",
    )


# ==============================================================================
# Tab 2: Explorador de potencial
# ==============================================================================

class TabExplorador(QWidget):
    """Pestaña de exploración interactiva del potencial efectivo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel de deslizadores
        panel_sliders = QWidget()
        lay_sl = QVBoxLayout(panel_sliders)
        panel_sliders.setMaximumWidth(280)

        lay_sl.addWidget(QLabel("Tipo de partícula:"))
        self.rb_masiva = QRadioButton("Masiva (ε=1)")
        self.rb_foton  = QRadioButton("Fotón (ε=0)")
        self.rb_masiva.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_masiva)
        bg.addButton(self.rb_foton)
        lay_sl.addWidget(self.rb_masiva)
        lay_sl.addWidget(self.rb_foton)

        lay_sl.addWidget(self._separador())

        # Control E: deslizador + spinbox
        lay_sl.addWidget(QLabel("Energía E:"))
        row_E = QHBoxLayout()
        self.slider_E = QSlider(Qt.Orientation.Horizontal)
        self.slider_E.setMinimum(0)
        self.slider_E.setMaximum(3000)
        self.slider_E.setValue(940)
        self.spin_E = QDoubleSpinBox()
        self.spin_E.setRange(0.0, 3.0)
        self.spin_E.setSingleStep(0.005)
        self.spin_E.setDecimals(3)
        self.spin_E.setValue(0.940)
        self.spin_E.setFixedWidth(80)
        row_E.addWidget(self.slider_E)
        row_E.addWidget(self.spin_E)
        lay_sl.addLayout(row_E)

        # Control L: deslizador + spinbox
        lay_sl.addWidget(QLabel("Momento angular L:"))
        row_L = QHBoxLayout()
        self.slider_L = QSlider(Qt.Orientation.Horizontal)
        self.slider_L.setMinimum(-2000)
        self.slider_L.setMaximum(2000)
        self.slider_L.setValue(340)
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(-20.0, 20.0)
        self.spin_L.setSingleStep(0.05)
        self.spin_L.setDecimals(3)
        self.spin_L.setValue(3.40)
        self.spin_L.setFixedWidth(80)
        row_L.addWidget(self.slider_L)
        row_L.addWidget(self.spin_L)
        lay_sl.addLayout(row_L)

        lay_sl.addWidget(self._separador())

        # Deslizador rango del eje x
        lay_sl.addWidget(QLabel("r_max del gráfico:"))
        self.lbl_rmax = QLabel("r_max = 40.0 M")
        lay_sl.addWidget(self.lbl_rmax)
        self.slider_rmax = QSlider(Qt.Orientation.Horizontal)
        self.slider_rmax.setMinimum(5)
        self.slider_rmax.setMaximum(200)
        self.slider_rmax.setValue(40)
        lay_sl.addWidget(self.slider_rmax)

        lay_sl.addWidget(self._separador())

        # Anotaciones
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMaximumHeight(200)
        self.txt_info.setFont(QFont("Consolas", 8))
        self.txt_info.setStyleSheet(
            "background-color: #0D1117; color: #88FF88; border: 1px solid #333;"
        )
        lay_sl.addWidget(QLabel("Información:"))
        lay_sl.addWidget(self.txt_info)

        lay_sl.addStretch()

        # Canvas
        self.figura = Figure(figsize=(10, 7), tight_layout=True)
        self.figura.patch.set_facecolor(COLORES['fondo_fig'])
        self.ax = self.figura.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figura)
        toolbar = NavigationToolbar(self.canvas, self)

        self._xlim_orig = None
        self._ylim_orig = None

        self._btn_reset = QPushButton("⌂ Reset vista")
        self._btn_reset.setToolTip("Restaura los límites originales de la gráfica")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_vista)

        panel_viz = QWidget()
        lay_viz = QVBoxLayout(panel_viz)
        lay_viz.addWidget(toolbar)
        lay_viz.addWidget(self.canvas)
        lay_viz.addWidget(self._btn_reset)

        splitter.addWidget(panel_sliders)
        splitter.addWidget(panel_viz)
        splitter.setSizes([260, 800])
        layout.addWidget(splitter)

        # Conexiones
        self.slider_E.valueChanged.connect(self._on_slider_E_changed)
        self.spin_E.valueChanged.connect(self._on_spin_E_changed)
        self.slider_L.valueChanged.connect(self._on_slider_L_changed)
        self.spin_L.valueChanged.connect(self._on_spin_L_changed)
        self.slider_rmax.valueChanged.connect(self._actualizar)
        self.rb_masiva.toggled.connect(self._actualizar)

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._actualizar()

    def _separador(self):
        linea = QFrame()
        linea.setFrameShape(QFrame.Shape.HLine)
        linea.setStyleSheet("color: #333333;")
        return linea

    def _on_slider_E_changed(self, val):
        self.spin_E.blockSignals(True)
        self.spin_E.setValue(val / 1000.0)
        self.spin_E.blockSignals(False)
        self._actualizar()

    def _on_spin_E_changed(self, val):
        self.slider_E.blockSignals(True)
        self.slider_E.setValue(round(val * 1000))
        self.slider_E.blockSignals(False)
        self._actualizar()

    def _on_slider_L_changed(self, val):
        self.spin_L.blockSignals(True)
        self.spin_L.setValue(val / 100.0)
        self.spin_L.blockSignals(False)
        self._actualizar()

    def _on_spin_L_changed(self, val):
        self.slider_L.blockSignals(True)
        self.slider_L.setValue(round(val * 100))
        self.slider_L.blockSignals(False)
        self._actualizar()

    def _get_params(self):
        E = self.spin_E.value()
        L = self.spin_L.value()
        r_max = float(self.slider_rmax.value())
        eps = 0 if self.rb_foton.isChecked() else 1
        return E, L, eps, r_max

    def _actualizar(self):
        E, L, eps, r_max = self._get_params()
        self.lbl_rmax.setText(f"r_max = {r_max:.0f} M")

        graficar_potencial_explorador(self.ax, L, eps, E, r_min=2.05, r_max=r_max)
        self.canvas.draw_idle()
        self._xlim_orig = self.ax.get_xlim()
        self._ylim_orig = self.ax.get_ylim()
        self._btn_reset.setEnabled(True)

        # Información textual estructurada
        from potentials import encontrar_radios_circulares, encontrar_puntos_retorno
        radios = encontrar_radios_circulares(L, eps)
        puntos = encontrar_puntos_retorno(E, L, eps, r_max=300.0)

        emoji, titulo, explicacion, interpretacion = _calcular_estado_fisico(
            E, L, eps, radios, puntos
        )

        info = []

        # 1. Estado físico esperado (siempre primero)
        info.append(f"{emoji} ESTADO: {titulo}")
        info.append(explicacion)

        # 2. Parámetros numéricos
        info.append("\n— Parámetros —")
        info.append(f"E = {E:.4f}   L = {L:.4f}")
        if abs(E) > 1e-6:
            b = abs(L / E)
            if eps == 0:
                delta_rel = (b - B_CRITICO) / B_CRITICO
                info.append(
                    f"b = |L/E|  = {b:.4f}\n"
                    f"b_c = 3√3  = {B_CRITICO:.4f}\n"
                    f"Δb/b_c     = {delta_rel * 100:+.2f}%"
                )
            else:
                info.append(f"b = |L/E| = {b:.4f}")

        # 3. Estructura del potencial
        info.append("\n— Estructura del potencial —")
        if eps == 0:
            # Para fotones, la esfera de fotones r=3M es siempre una referencia
            # geométrica, no una órbita circular asociada a los parámetros actuales
            info.append("Esfera de fotones: r = 3M  (referencia geométrica)")
        else:
            if radios:
                r_est, r_inest = radios
                if r_est:
                    info.append(f"Circ. estable:   r = {r_est:.4f} M")
                if r_inest:
                    info.append(f"Circ. inestable: r = {r_inest:.4f} M")
            else:
                info.append("Sin órbitas circulares (L < 2√3 M)")

        if puntos:
            pts_str = ", ".join(f"{p:.3f}M" for p in puntos)
            info.append(f"Puntos de retorno: {pts_str}")
        else:
            info.append("Sin puntos de retorno")

        # 4. Interpretación física (modo docente)
        info.append("\n— Interpretación física —")
        info.append(interpretacion)

        self.txt_info.setPlainText("\n".join(info))

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        xc = event.xdata
        escala = 1.0 - _ZOOM_FACTOR if event.button == 'up' else 1.0 + _ZOOM_FACTOR
        x0, x1 = self.ax.get_xlim()
        r_min = max(xc - (xc - x0) * escala, 2.02)
        r_max = max(xc + (x1 - xc) * escala, r_min + 0.1)
        E, L, eps, _ = self._get_params()
        graficar_potencial_explorador(self.ax, L, eps, E, r_min=r_min, r_max=r_max)
        self.canvas.draw_idle()

    def _reset_vista(self):
        if self._xlim_orig is not None:
            # Redibujar desde cero con el rango original
            E, L, eps, r_max_sl = self._get_params()
            graficar_potencial_explorador(self.ax, L, eps, E, r_min=2.05, r_max=r_max_sl)
            self.canvas.draw_idle()

    def get_figura(self):
        return self.figura


# ==============================================================================
# Tab 3: Comparador de trayectorias
# ==============================================================================

class TabComparador(QWidget):
    """Pestaña para comparar varias trayectorias simultáneamente."""

    COLORES_DISPONIBLES = [
        "#FF4444", "#FF9900", "#FFFF44", "#44FF44",
        "#44AAFF", "#FF44FF", "#44FFFF", "#FFFFFF",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel izquierdo: configuración de trayectorias
        panel_izq = QWidget()
        lay_izq = QVBoxLayout(panel_izq)
        panel_izq.setMaximumWidth(300)

        lay_izq.addWidget(QLabel("Configuración rápida:"))
        btn_cargar_bc = QPushButton("Cargar comparación b vs b_c")
        btn_cargar_bc.clicked.connect(self._cargar_comparacion_bc)
        lay_izq.addWidget(btn_cargar_bc)

        btn_cargar_defl = QPushButton("Cargar deflexión fotones")
        btn_cargar_defl.setToolTip(
            "Carga 7 fotones de dispersión con b = 6–25 M\n"
            "para comparar el ángulo de deflexión gravitacional."
        )
        btn_cargar_defl.clicked.connect(self._cargar_deflexion_foton)
        lay_izq.addWidget(btn_cargar_defl)

        lay_izq.addWidget(self._separador())

        # Formulario para añadir trayectoria manual
        lay_izq.addWidget(QLabel("Añadir trayectoria:"))

        grp_add = QGroupBox()
        lay_add = QGridLayout(grp_add)

        self.add_rb_masiva = QRadioButton("Masiva")
        self.add_rb_foton  = QRadioButton("Fotón")
        self.add_rb_masiva.setChecked(True)
        bg2 = QButtonGroup(self)
        bg2.addButton(self.add_rb_masiva)
        bg2.addButton(self.add_rb_foton)
        lay_add.addWidget(self.add_rb_masiva, 0, 0)
        lay_add.addWidget(self.add_rb_foton, 0, 1)

        self.add_spin_E  = self._make_spin(0.0, 5.0, 1.0, 0.01)
        self.add_spin_L  = self._make_spin(-20.0, 20.0, B_CRITICO * 1.2, 0.1)
        self.add_spin_r0 = self._make_spin(2.01, 200.0, 40.0, 1.0)
        self.add_spin_lam = self._make_spin(10, 5000, 400, 50)
        self.add_spin_lam.setDecimals(0)

        lay_add.addWidget(QLabel("E"), 1, 0)
        lay_add.addWidget(self.add_spin_E, 1, 1)
        lay_add.addWidget(QLabel("L"), 2, 0)
        lay_add.addWidget(self.add_spin_L, 2, 1)
        lay_add.addWidget(QLabel("r₀"), 3, 0)
        lay_add.addWidget(self.add_spin_r0, 3, 1)
        lay_add.addWidget(QLabel("λ_max"), 4, 0)
        lay_add.addWidget(self.add_spin_lam, 4, 1)

        btn_agregar = QPushButton("+ Añadir")
        btn_agregar.clicked.connect(self._agregar_trayectoria)
        lay_add.addWidget(btn_agregar, 5, 0, 1, 2)
        lay_izq.addWidget(grp_add)

        lay_izq.addWidget(self._separador())

        # Lista de trayectorias
        lay_izq.addWidget(QLabel("Trayectorias activas:"))
        self.lista_tray = QListWidget()
        self.lista_tray.setMaximumHeight(200)
        lay_izq.addWidget(self.lista_tray)

        btn_eliminar = QPushButton("✕ Eliminar seleccionada")
        btn_eliminar.clicked.connect(self._eliminar_seleccionada)
        lay_izq.addWidget(btn_eliminar)

        btn_limpiar = QPushButton("✕ Limpiar todo")
        btn_limpiar.clicked.connect(self._limpiar_todo)
        lay_izq.addWidget(btn_limpiar)

        btn_graficar = QPushButton("▶ Graficar todas")
        btn_graficar.setMinimumHeight(36)
        btn_graficar.clicked.connect(self._graficar_todas)
        lay_izq.addWidget(btn_graficar)

        lay_izq.addStretch()

        # Canvas
        self.figura = Figure(figsize=(10, 8), tight_layout=True)
        self.figura.patch.set_facecolor(COLORES['fondo_fig'])
        self.ax = self.figura.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figura)
        toolbar = NavigationToolbar(self.canvas, self)

        self._xlim_orig = None
        self._ylim_orig = None

        self._btn_reset = QPushButton("⌂ Reset vista")
        self._btn_reset.setToolTip("Restaura los límites originales de la gráfica")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_vista)

        panel_viz = QWidget()
        lay_viz = QVBoxLayout(panel_viz)
        lay_viz.addWidget(toolbar)
        lay_viz.addWidget(self.canvas)
        lay_viz.addWidget(self._btn_reset)

        splitter.addWidget(panel_izq)
        splitter.addWidget(panel_viz)
        splitter.setSizes([280, 820])
        layout.addWidget(splitter)

        # Almacenamiento de configs y últimos resultados integrados
        self.configs_tray = []
        self.resultados_actuales = []
        self.idx_color = 0
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _separador(self):
        linea = QFrame()
        linea.setFrameShape(QFrame.Shape.HLine)
        linea.setStyleSheet("color: #333333;")
        return linea

    def _make_spin(self, minv, maxv, val, step):
        sp = QDoubleSpinBox()
        sp.setRange(minv, maxv)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setDecimals(4)
        return sp

    def _cargar_comparacion_bc(self):
        self.configs_tray = trayectorias_comparacion_bc()
        self.lista_tray.clear()
        for cfg in self.configs_tray:
            item = QListWidgetItem(cfg['etiqueta'])
            item.setForeground(QColor(cfg['color']))
            self.lista_tray.addItem(item)
        self._graficar_todas()

    def _cargar_deflexion_foton(self):
        self.configs_tray = trayectorias_deflexion_foton()
        self.lista_tray.clear()
        for cfg in self.configs_tray:
            item = QListWidgetItem(cfg['etiqueta'])
            item.setForeground(QColor(cfg['color']))
            self.lista_tray.addItem(item)
        self._graficar_todas()

    def _agregar_trayectoria(self):
        eps = 0 if self.add_rb_foton.isChecked() else 1
        E   = self.add_spin_E.value()
        L   = self.add_spin_L.value()
        r0  = self.add_spin_r0.value()
        lam = self.add_spin_lam.value()
        color = self.COLORES_DISPONIBLES[self.idx_color % len(self.COLORES_DISPONIBLES)]
        self.idx_color += 1

        b = abs(L / E) if abs(E) > 1e-6 else float('inf')
        etiqueta = f"{'F' if eps==0 else 'M'} E={E:.3f} L={L:.3f} b={b:.3f}"

        cfg = {
            "E": E, "L": L, "r0": r0, "phi0": np.pi,
            "rdot0": None, "sentido": -1, "epsilon": eps,
            "lambda_max": lam, "etiqueta": etiqueta, "color": color,
        }
        self.configs_tray.append(cfg)
        item = QListWidgetItem(etiqueta)
        item.setForeground(QColor(color))
        self.lista_tray.addItem(item)

    def _eliminar_seleccionada(self):
        fila = self.lista_tray.currentRow()
        if 0 <= fila < len(self.configs_tray):
            self.configs_tray.pop(fila)
            self.lista_tray.takeItem(fila)

    def _limpiar_todo(self):
        self.configs_tray.clear()
        self.resultados_actuales.clear()
        self.lista_tray.clear()
        self.ax.cla()
        self.canvas.draw_idle()
        self._xlim_orig = None
        self._ylim_orig = None
        self._btn_reset.setEnabled(False)

    def _graficar_todas(self):
        if not self.configs_tray:
            return

        resultados = []
        for cfg in self.configs_tray:
            params = {k: v for k, v in cfg.items()
                      if k not in ('etiqueta', 'color')}
            res = integrar_geodesica(**params)
            resultados.append(res)

        self.resultados_actuales = resultados
        graficar_comparacion(self.ax, resultados, self.configs_tray)
        self.canvas.draw_idle()
        self._xlim_orig = self.ax.get_xlim()
        self._ylim_orig = self.ax.get_ylim()
        self._btn_reset.setEnabled(True)

    def _on_scroll(self, event):
        _zoom_scroll(event, self.ax, es_aspect_equal=True)

    def _reset_vista(self):
        if self._xlim_orig is not None:
            self.ax.set_xlim(self._xlim_orig)
            self.ax.set_ylim(self._ylim_orig)
            self.ax.set_aspect('equal', adjustable='datalim')
            self.canvas.draw_idle()

    def get_figura(self):
        return self.figura

    def get_resultados(self):
        return self.resultados_actuales


# ==============================================================================
# ==============================================================================
# Exportación de metadatos junto a la figura
# ==============================================================================

def _guardar_metadatos_txt(resultado, ruta, metodo='DOP853', tolerancia='1e-10', lambda_max=None):
    """
    Guarda un archivo .txt con los metadatos de la simulación.

    Incluye: E, L, epsilon, r0, phi0, rdot0, tipo de trayectoria (teórica
    y numérica), r_min, r_max, lambda_total y conservación de energía máxima.
    También incluye la advertencia de rdot0 manual si existe.

    Bloques cuantitativos adicionales (cuando aplican):
        - PRECESIÓN RELATIVISTA DEL PERIASTRO (órbitas ligadas con precesión)
        - DILATACIÓN TEMPORAL INTEGRADA (partículas masivas)
        - VALIDACIÓN NUMÉRICA (siempre que haya datos de residuo)
    """
    from integrator import (residuo_conservacion,
                             calcular_precesion_perihelio,
                             calcular_tiempo_propio,
                             calcular_angulo_deflexion)
    from datetime import datetime

    E   = resultado['E']
    L   = resultado['L']
    eps = resultado['epsilon']
    r0    = float(resultado['r'][0])   if len(resultado['r'])    > 0 else float('nan')
    phi0  = float(resultado['phi'][0]) if len(resultado['phi'])  > 0 else float('nan')
    rdot0 = float(resultado['rdot'][0])if len(resultado['rdot']) > 0 else float('nan')
    tipo_num  = resultado['tipo']
    r_min     = resultado.get('r_min', float('nan'))
    r_max_val = resultado.get('r_max', float('nan'))
    lam_total = float(resultado['lam'][-1]) if len(resultado['lam']) > 0 else 0.0

    # Clasificación teórica detallada (ISCO, circular estable/inestable, etc.)
    tipo_teo = clasificar_trayectoria_teorica(E, L, eps, r0=r0)

    # Conservación de energía a lo largo de la trayectoria (reutilizada en el bloque 3)
    _, delta = residuo_conservacion(resultado)
    conserv_max = float(np.max(delta)) if len(delta) > 0 else float('nan')

    tipo_part = "foton (epsilon=0)" if eps == 0 else "masiva (epsilon=1)"
    tipo_part_corto = "Fotón" if eps == 0 else "Partícula masiva"

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    lam_max_val = lambda_max if lambda_max is not None else lam_total

    lineas = [
        "====================================================",
        "SIMULADOR DE GEODÉSICAS DE SCHWARZSCHILD",
        "Versión: 1.0",
        f"Fecha: {fecha}",
        f"Método: {metodo}",
        f"Tolerancia: {tolerancia}",
        f"λ_max: {lam_max_val:.0f}",
        "r_max: 300",
        f"Tipo: {tipo_part_corto}",
        f"E: {E:.4f}",
        f"L: {L:.4f}",
        f"r0: {r0:.4f}",
        "====================================================",
        "",
        "# Metadatos de simulacion — Geodesicas de Schwarzschild",
        "# Unidades geometrizadas: G = c = M = 1",
        "",
        f"E                  = {E:.8f}",
        f"L                  = {L:.8f}",
        f"epsilon            = {eps}  ({tipo_part})",
        f"r0                 = {r0:.8f} M",
        f"phi0               = {phi0:.8f} rad",
        f"rdot0              = {rdot0:.8e}",
        "",
        f"Tipo trayectoria (teoria)   : {tipo_teo}",
        f"Tipo trayectoria (integrador): {tipo_num}",
        "",
        f"r_min              = {r_min:.6f} M",
        f"r_max              = {r_max_val:.6f} M",
        f"lambda_total       = {lam_total:.4f}",
        "",
        f"Conservacion energia max|E^2-Veff-rdot^2| = {conserv_max:.3e}",
    ]

    # Incluir advertencia de rdot0 manual si procede
    aviso = resultado.get('aviso_rdot0')
    if aviso:
        lineas += ["", aviso]

    # ── BLOQUE 1: PRECESIÓN RELATIVISTA DEL PERIASTRO ────────────────────────
    prec = calcular_precesion_perihelio(resultado)
    if prec is not None:
        discrepancia = (abs(prec['exceso_num_rad'] - prec['exceso_anal_rad'])
                        / abs(prec['exceso_anal_rad']) * 100)
        p_val = prec['semi_latus_rectum']
        e_val = prec['excentricidad']
        # Comentario automático según el régimen de campo
        if p_val < 20.0:
            nota_regimen = (
                f"Regimen relativista fuerte (p = {p_val:.1f} M, "
                f"r_peri = {prec['r_peri']:.1f} M ~ ISCO). "
                "La formula 6pi/p es una aproximacion perturbativa de campo debil "
                "valida solo para p >> 6M. La discrepancia elevada es esperable "
                "para orbitas con periastro cercano al radio ISCO (6M) y no "
                "refleja ningún error del integrador numerico."
            )
        elif e_val < 0.25:
            nota_regimen = (
                f"Regimen de campo debil (p = {p_val:.1f} M, e = {e_val:.3f}). "
                "La formula 6pi/p es una buena aproximacion a primer orden. "
                "La discrepancia residual refleja correcciones de orden superior "
                "en M/p no capturadas por la formula analitica de primer orden."
            )
        else:
            nota_regimen = (
                f"Regimen intermedio (p = {p_val:.1f} M, e = {e_val:.3f}). "
                "La formula 6pi/p da una estimacion aproximada; contribuyen "
                "correcciones de orden superior tanto en M/p como en e."
            )
        lineas += [
            "",
            "====================================================",
            "PRECESIÓN RELATIVISTA DEL PERIASTRO",
            "====================================================",
            f"Periapsis detectados           : {prec['n_periapsis']}",
            f"r_peri medio                   = {prec['r_peri']:.6f} M",
            f"r_apo medio                    = {prec['r_apo']:.6f} M",
            f"Semi-latus rectum p            = {prec['semi_latus_rectum']:.6f} M",
            f"Excentricidad e                = {prec['excentricidad']:.6f}",
            f"Precesión numérica por órbita  = {prec['exceso_num_rad']:.6f} rad",
            f"Precesión analítica (6π/p)     = {prec['exceso_anal_rad']:.6f} rad",
            f"Discrepancia relativa          = {discrepancia:.2f}%",
            f"Interpretacion: {nota_regimen}",
        ]

    # ── BLOQUE 2: DILATACIÓN TEMPORAL CUANTITATIVA ───────────────────────────
    tp = calcular_tiempo_propio(resultado)
    if tp is not None:
        lineas += [
            "",
            "====================================================",
            "DILATACIÓN TEMPORAL INTEGRADA",
            "====================================================",
            f"Tiempo propio total τ          = {tp['tau']:.6f} M",
            f"Tiempo coordenado total t      = {tp['t_coo']:.6f} M",
            f"Cociente τ/t                   = {tp['ratio']:.8f}",
            f"Diferencia t − τ               = {tp['atraso']:.6f} M",
        ]

    # ── BLOQUE 3: ANÁLISIS TEMPORAL DIFERENCIAL ───────────────────────────────
    r_traj = resultado['r']
    if len(r_traj) > 1:
        fr_traj = 1.0 - 2.0 / r_traj
        if eps == 1:
            # Partícula masiva: calcular ambos factores
            fe_arr = np.where(fr_traj > 0, np.sqrt(fr_traj), np.nan)
            fp_arr = np.where(fr_traj > 0, fr_traj / E,      np.nan)
            fe_min  = float(np.nanmin(fe_arr))
            fe_max  = float(np.nanmax(fe_arr))
            fe_mean = float(np.nanmean(fe_arr))
            fp_min  = float(np.nanmin(fp_arr))
            fp_max  = float(np.nanmax(fp_arr))
            fp_mean = float(np.nanmean(fp_arr))
            lineas += [
                "",
                "====================================================",
                "ANÁLISIS TEMPORAL",
                "====================================================",
                "Factor estatico  dtau_est/dt = sqrt(1 - 2M/r(tau)):",
                f"  min  = {fe_min:.6f}",
                f"  max  = {fe_max:.6f}",
                f"  media= {fe_mean:.6f}",
                "",
                "Factor propio    dtau_part/dt = (1 - 2M/r(tau)) / E:",
                f"  min  = {fp_min:.6f}",
                f"  max  = {fp_max:.6f}",
                f"  media= {fp_mean:.6f}",
                "",
                "Nota: el factor propio (dtau_part/dt) es siempre menor o igual",
                "al factor estatico, lo que refleja la dilatacion temporal",
                "adicional por movimiento de la particula.",
            ]
        else:
            # Fotón: tiempo propio no definido
            lineas += [
                "",
                "====================================================",
                "ANÁLISIS TEMPORAL",
                "====================================================",
                "Dilatacion temporal propia:",
                "No aplicable. Las geodesicas nulas no poseen tiempo propio",
                "asociado al foton.",
            ]

    # ── BLOQUE 4: ÁNGULO DE DEFLEXIÓN (fotones de dispersión) ────────────────
    defl = calcular_angulo_deflexion(resultado)
    if defl is not None:
        # Indicador de régimen según b
        if defl['b'] < 8.0:
            nota_defl = (
                f"Regimen de campo fuerte (b = {defl['b']:.2f} M, "
                f"r_min = {defl['r_min']:.2f} M). "
                "La formula 4M/b subestima la deflexion real; contribuyen "
                "correcciones de orden superior significativas."
            )
        elif defl['b'] < 15.0:
            nota_defl = (
                f"Regimen intermedio (b = {defl['b']:.2f} M). "
                "La formula 4M/b es una aproximacion razonable."
            )
        else:
            nota_defl = (
                f"Regimen de campo debil (b = {defl['b']:.2f} M). "
                "La formula 4M/b converge bien con α_corr. "
                "Nota: α_sim puede ser negativo si b ~ r0 (radio inicial finito); "
                "usar α_corr (extrapolado a ∞) para la comparacion analitica."
            )
        lineas += [
            "",
            "====================================================",
            "ÁNGULO DE DEFLEXIÓN GRAVITACIONAL",
            "====================================================",
            f"Parámetro de impacto b         = {defl['b']:.6f} M",
            f"r mínimo alcanzado             = {defl['r_min']:.6f} M",
            f"r inicio sim. / r final sim.   = {defl['r0_sim']:.1f} M / {defl['r_final_sim']:.1f} M",
            f"|Δφ| barrido (simulación)      = {defl['delta_phi_sim']:.6f} rad",
            f"α = |Δφ| − π  (sim.)          = {defl['alpha_rad']:.6f} rad  = {defl['alpha_deg']:.4f} °",
            f"α_corr (extrapolado a ∞)       = {defl['alpha_corr_rad']:.6f} rad  = {defl['alpha_corr_deg']:.4f} °",
            f"α_WF  = 4M/b  (campo débil)   = {defl['alpha_WF_rad']:.6f} rad  = {defl['alpha_WF_deg']:.4f} °",
            f"Interpretación: {nota_defl}",
        ]
    elif eps == 0:
        lineas += [
            "",
            "====================================================",
            "ÁNGULO DE DEFLEXIÓN GRAVITACIONAL",
            "====================================================",
            "No aplicable: trayectoria no es de dispersión con periastro claro.",
            "(Captura, esfera de fotones o fotón sin punto de retorno radial.)",
        ]

    # ── BLOQUE 5: VALIDACIÓN NUMÉRICA ─────────────────────────────────────────
    if len(delta) > 0:
        resid_max  = float(np.max(delta))
        resid_mean = float(np.mean(delta))
        if resid_max < 1e-9:
            comentario = ("La conservación de la energía es excelente y confirma "
                          "la estabilidad numérica de la integración.")
        elif resid_max < 1e-6:
            comentario = ("La conservación de la energía es adecuada para la "
                          "interpretación cualitativa.")
        else:
            comentario = ("La simulación debe revisarse o repetirse con tolerancia "
                          "más estricta.")
        lineas += [
            "",
            "====================================================",
            "VALIDACIÓN NUMÉRICA",
            "====================================================",
            f"Residuo máximo  |E²−V_eff−ṙ²|  = {resid_max:.3e}",
            f"Residuo medio   |E²−V_eff−ṙ²|  = {resid_mean:.3e}",
            f"Calidad numérica: {comentario}",
        ]

    with open(ruta, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lineas) + '\n')


def _guardar_metadatos_comparador_txt(configs, resultados, ruta, nombre=""):
    """
    Genera el TXT de metadatos para una exportación del comparador.

    Parámetros
    ----------
    configs    : lista de dicts de configuración (uno por trayectoria)
    resultados : lista de dicts devueltos por integrar_geodesica
    ruta       : ruta completa del archivo .txt a escribir
    nombre     : nombre descriptivo de la comparación
    """
    from datetime import datetime

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(configs)

    lineas = [
        "====================================================",
        "COMPARADOR DE TRAYECTORIAS — GEODÉSICAS DE SCHWARZSCHILD",
        f"Nombre     : {nombre or 'Comparación sin título'}",
        f"Fecha      : {fecha}",
        "Masa       : M = 1  (unidades geométricas  G = c = M = 1)",
        f"Trayectorias: {n}",
        "====================================================",
        "",
    ]

    hay_critica = False

    for i, (cfg, res) in enumerate(zip(configs, resultados), start=1):
        eps      = cfg.get('epsilon', 1)
        E        = float(cfg.get('E', float('nan')))
        L        = float(cfg.get('L', float('nan')))
        r0       = float(cfg.get('r0', float('nan')))
        etiqueta = cfg.get('etiqueta', f"Trayectoria {i}")
        tipo_num = res.get('tipo', '—')
        tipo_part = "Fotón  (ε=0)" if eps == 0 else "Masiva (ε=1)"

        if abs(E) > 1e-10:
            b = abs(L / E)
            b_str = f"{b:.4f}"
            if eps == 0 and abs(b - B_CRITICO) / B_CRITICO < 0.05:
                hay_critica = True
        else:
            b_str = "—  (E ≈ 0)"

        lineas += [
            f"── Trayectoria {i}:  {etiqueta}",
            f"   Tipo partícula  : {tipo_part}",
            f"   E               = {E:.6f}",
            f"   L               = {L:.6f}",
            f"   b = |L/E|       = {b_str}",
            f"   Radio inicial   = {r0:.4f} M",
            f"   Resultado       = {tipo_num}",
            "",
        ]

    if hay_critica:
        lineas += [
            "── Parámetro crítico de fotón",
            f"   b_c = 3√3       = {B_CRITICO:.6f}",
            "",
        ]

    # ── TABLA ÁNGULO DE DEFLEXIÓN (fotones de dispersión) ──────────────────────
    from integrator import calcular_angulo_deflexion
    filas_defl = []
    for cfg, res in zip(configs, resultados):
        if cfg.get('epsilon', 1) != 0:
            continue
        defl = calcular_angulo_deflexion(res)
        E_val = float(cfg.get('E', 1.0))
        L_val = float(cfg.get('L', 0.0))
        b_val = abs(L_val / E_val) if abs(E_val) > 1e-10 else float('inf')
        etiqueta = cfg.get('etiqueta', '—')
        tipo_res = res.get('tipo', '—')
        if defl is not None:
            filas_defl.append((
                b_val,
                defl['alpha_rad'],
                defl['alpha_deg'],
                defl['alpha_corr_rad'],
                defl['alpha_corr_deg'],
                defl['alpha_WF_rad'],
                defl['r_min'],
                tipo_res,
                etiqueta,
            ))
        elif cfg.get('epsilon', 1) == 0:
            filas_defl.append((b_val, None, None, None, None, None, None, tipo_res, etiqueta))

    if filas_defl:
        lineas += [
            "── TABLA: ÁNGULO DE DEFLEXIÓN GRAVITACIONAL (fotones)",
            f"   b_c = 3√3 ≈ {B_CRITICO:.4f} M",
            "",
            f"   {'b/M':>7}  {'α_sim(rad)':>11}  {'α_sim(°)':>9}  "
            f"{'α_corr(rad)':>12}  {'α_corr(°)':>10}  "
            f"{'α_WF(rad)':>10}  {'r_min/M':>7}  resultado",
        ]
        for fila in filas_defl:
            b_v, a_r, a_d, ac_r, ac_d, aw_r, rm, tipo_r, _ = fila
            if a_r is not None:
                lineas.append(
                    f"   {b_v:>7.3f}  {a_r:>11.6f}  {a_d:>9.4f}  "
                    f"{ac_r:>12.6f}  {ac_d:>10.4f}  "
                    f"{aw_r:>10.6f}  {rm:>7.4f}  {tipo_r}"
                )
            else:
                lineas.append(
                    f"   {b_v:>7.3f}  {'N/A':>11}  {'N/A':>9}  "
                    f"{'N/A':>12}  {'N/A':>10}  {'N/A':>10}  {'—':>7}  {tipo_r}"
                )
        lineas.append("")

    with open(ruta, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lineas))


# ==============================================================================
# Utilidades del sistema de exportación
# ==============================================================================

def _sanitizar_nombre(nombre):
    """Elimina caracteres no válidos en nombres de archivo (Windows/POSIX)."""
    import re
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', nombre)
    nombre = nombre.replace(' ', '_')
    nombre = nombre.strip('._')
    return nombre or 'exportacion'


def _generar_nombre_auto(tab_idx, resultado=None):
    """Nombre descriptivo basado en pestaña activa, tipo, E, L y fecha/hora."""
    from datetime import datetime
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefijos = ["simulador", "explorador", "comparador", "referencia_estatica"]
    prefijo = prefijos[tab_idx]

    if resultado is not None:
        eps  = resultado.get('epsilon', 1)
        E    = resultado.get('E', 0.0)
        L    = resultado.get('L', 0.0)
        tipo = 'foton' if eps == 0 else 'masiva'
        E_s  = ('m' if E < 0 else '') + f"{abs(E):.3f}".replace('.', 'p')
        L_s  = ('m' if L < 0 else '') + f"{abs(L):.3f}".replace('.', 'p')
        return f"{prefijo}_{tipo}_E{E_s}_L{L_s}_{fecha}"
    return f"{prefijo}_{fecha}"


def _resolver_sin_colision(directorio, nombre_base):
    """
    Devuelve (ruta_jpg, ruta_pdf, ruta_txt) libres de colisión.
    Si alguno de los archivos ya existe añade sufijo _01, _02 … _99.
    """
    def _rutas(nb):
        return (
            os.path.join(directorio, nb + '.jpg'),
            os.path.join(directorio, nb + '.pdf'),
            os.path.join(directorio, nb + '.txt'),
        )

    terna = _rutas(nombre_base)
    if not any(os.path.exists(p) for p in terna):
        return terna

    for i in range(1, 100):
        terna = _rutas(f"{nombre_base}_{i:02d}")
        if not any(os.path.exists(p) for p in terna):
            return terna

    from datetime import datetime
    return _rutas(f"{nombre_base}_{datetime.now().strftime('%H%M%S')}")


def _guardar_jpg_pdf(fig, ruta_jpg, ruta_pdf):
    """Guarda la figura como JPG de alta calidad (95) y PDF vectorial.

    El JPG se compuesta contra fondo blanco para eliminar el canal alfa,
    ya que JPEG no admite transparencia. Propaga cualquier excepción para
    que el llamador la muestre con QMessageBox y decida si continúa.
    """
    os.makedirs(os.path.dirname(ruta_jpg) or '.', exist_ok=True)
    fc = fig.get_facecolor()          # siempre (r, g, b, a) en [0, 1]
    r, g, b, a = float(fc[0]), float(fc[1]), float(fc[2]), float(fc[3])
    # Compositar sobre blanco para obtener color opaco válido para JPEG
    fc_opaque = (a * r + (1.0 - a),
                 a * g + (1.0 - a),
                 a * b + (1.0 - a))
    fig.savefig(ruta_jpg, dpi=200, bbox_inches='tight',
                facecolor=fc_opaque, format='jpeg',
                pil_kwargs={'quality': 95, 'optimize': True})
    fig.savefig(ruta_pdf, bbox_inches='tight', facecolor=fc)


def _guardar_solo_jpg(fig, ruta, dpi=300):
    """Guarda la figura como JPG de alta resolución (dpi=300) sin generar PDF."""
    fc = fig.get_facecolor()
    r, g, b, a = float(fc[0]), float(fc[1]), float(fc[2]), float(fc[3])
    fc_opaque = (a * r + (1.0 - a), a * g + (1.0 - a), a * b + (1.0 - a))
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    fig.savefig(ruta, dpi=dpi, bbox_inches='tight',
                facecolor=fc_opaque, format='jpeg',
                pil_kwargs={'quality': 95, 'optimize': True})


# ==============================================================================
# Utilidad de zoom con rueda del ratón (reutilizada en varios paneles)
# ==============================================================================

_ZOOM_FACTOR = 0.15   # fracción del rango ganada/perdida por paso de rueda


def _zoom_scroll(event, ax, es_aspect_equal=False):
    """
    Aplica zoom centrado en el cursor al recibir un scroll_event de matplotlib.
    Modifica ax.set_xlim() / ax.set_ylim() directamente y llama a draw_idle().
    """
    if event.inaxes is not ax:
        return
    xc, yc = event.xdata, event.ydata
    if xc is None or yc is None:
        return

    escala = 1.0 - _ZOOM_FACTOR if event.button == 'up' else 1.0 + _ZOOM_FACTOR

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    ax.set_xlim(xc - (xc - x0) * escala, xc + (x1 - xc) * escala)
    ax.set_ylim(yc - (yc - y0) * escala, yc + (y1 - yc) * escala)

    if es_aspect_equal:
        ax.set_aspect('equal', adjustable='datalim')

    ax.get_figure().canvas.draw_idle()


# ==============================================================================
# Ventana ampliada al hacer doble clic sobre un panel
# ==============================================================================

class VentanaAmpliadaPanel(QDialog):
    """Muestra uno de los cuatro paneles del simulador en una ventana grande."""

    _TITULOS = [
        "Trayectoria orbital — ampliada",
        "Potencial efectivo — ampliado",
        "Evolución radial r(λ) — ampliada",
        "Dilatación temporal de la trayectoria — ampliada",
    ]

    def __init__(self, panel_idx, resultado, M_solar, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self._TITULOS[panel_idx])
        self.resize(950, 750)

        # panel_idx 0 = trayectoria (aspect equal), resto = libre
        self._panel_idx    = panel_idx
        self._es_orbital   = (panel_idx == 0)
        self._resultado_amp = resultado   # necesario para redibujar en zoom

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor(COLORES['fondo_fig'])
        self._ax = self._fig.add_subplot(1, 1, 1)

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        toolbar = NavigationToolbar(self._canvas, self)

        E       = resultado['E']
        L       = resultado['L']
        epsilon = resultado['epsilon']

        if panel_idx == 0:
            graficar_trayectoria(self._ax, resultado, M_solar=M_solar)
        elif panel_idx == 1:
            graficar_potencial(self._ax, L, epsilon, E)
        elif panel_idx == 2:
            graficar_evolucion_radial(self._ax, resultado)
        else:
            graficar_dilatacion_temporal(self._ax, resultado)

        self._fig.tight_layout()
        self._canvas.draw()

        # Guardar límites originales para el botón Reset
        self._xlim0 = self._ax.get_xlim()
        self._ylim0 = self._ax.get_ylim()

        # Zoom con rueda del ratón
        self._canvas.mpl_connect('scroll_event', self._on_scroll)

        # Fila inferior: Reset vista + Cerrar
        fila_btns = QHBoxLayout()
        btn_reset = QPushButton("⌂ Reset vista")
        btn_reset.setToolTip("Restaura los límites originales de la gráfica")
        btn_reset.clicked.connect(self._reset_vista)
        btn_cerrar = QPushButton("Cerrar")
        btn_cerrar.clicked.connect(self.accept)
        fila_btns.addWidget(btn_reset)
        fila_btns.addStretch()
        fila_btns.addWidget(btn_cerrar)

        layout.addWidget(toolbar)
        layout.addWidget(self._canvas)

        # Controles de animación — solo para la trayectoria orbital
        if panel_idx == 0:
            r   = resultado['r']
            phi = resultado['phi']
            self._anim_x        = r * np.cos(phi)
            self._anim_y        = r * np.sin(phi)
            self._anim_n        = len(r)
            self._anim_frame    = 0
            self._anim_idx_vel  = 0
            self._anim_marcador = None

            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._sig_frame_amp)

            fila_anim = QHBoxLayout()
            self._btn_play_amp  = QPushButton("▶ Play")
            self._btn_pausa_amp = QPushButton("⏸ Pausa")
            self._btn_reset_amp = QPushButton("⏹ Reset")
            for btn in [self._btn_play_amp, self._btn_pausa_amp, self._btn_reset_amp]:
                btn.setMaximumWidth(80)
                fila_anim.addWidget(btn)
            self._btn_play_amp.clicked.connect(self._play_amp)
            self._btn_pausa_amp.clicked.connect(self._pausa_amp)
            self._btn_reset_amp.clicked.connect(self._reiniciar_amp)

            fila_anim.addSpacing(8)
            fila_anim.addWidget(QLabel("Velocidad:"))
            self._slider_vel_amp = QSlider(Qt.Orientation.Horizontal)
            self._slider_vel_amp.setMinimum(0)
            self._slider_vel_amp.setMaximum(5)
            self._slider_vel_amp.setValue(0)
            self._slider_vel_amp.setMaximumWidth(130)
            self._slider_vel_amp.setTickPosition(QSlider.TickPosition.TicksBelow)
            self._lbl_vel_amp = QLabel("×1")
            self._lbl_vel_amp.setMinimumWidth(28)
            self._slider_vel_amp.valueChanged.connect(self._cambiar_vel_amp)
            fila_anim.addWidget(self._slider_vel_amp)
            fila_anim.addWidget(self._lbl_vel_amp)
            fila_anim.addStretch()

            layout.addLayout(fila_anim)

        layout.addLayout(fila_btns)

    def _on_scroll(self, event):
        _zoom_scroll(event, self._ax, es_aspect_equal=self._es_orbital)
        if self._panel_idx in (1, 3):
            x0, x1 = self._ax.get_xlim()
            self._redibujar_funcion(max(x0, 2.02), max(x1, 2.02 + 0.1))

    def _redibujar_funcion(self, r_min, r_max):
        """Redibuja la curva de función de r con el rango [r_min, r_max]."""
        res = self._resultado_amp
        if self._panel_idx == 1:
            graficar_potencial(self._ax, res['L'], res['epsilon'], res['E'],
                               r_min=r_min, r_max=r_max)
        elif self._panel_idx == 3:
            graficar_dilatacion_temporal(self._ax, res, r_min=r_min, r_max=r_max)
        self._canvas.draw_idle()

    def _reset_vista(self):
        if self._panel_idx in (1, 3):
            # Redibujar con el rango original garantiza curva + límites correctos
            r0, r1 = self._xlim0
            self._redibujar_funcion(max(r0, 2.02), max(r1, 2.02 + 0.1))
        else:
            self._ax.set_xlim(self._xlim0)
            self._ax.set_ylim(self._ylim0)
            if self._es_orbital:
                self._ax.set_aspect('equal', adjustable='datalim')
            self._canvas.draw_idle()

    # ----- Animación orbital (solo panel_idx == 0) -----

    _VELOCIDADES_AMP = [1, 2, 5, 10, 20, 50]

    def _play_amp(self):
        if self._anim_marcador is None:
            self._anim_marcador, = self._ax.plot(
                [], [], 'o', color=COLORES['marcador'],
                markersize=12, zorder=20, label='Partícula')
            self._ax.legend(loc='upper right', fontsize=8, ncol=2)
        self._anim_frame = 0
        self._anim_timer.start(50)

    def _pausa_amp(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        else:
            self._anim_timer.start(50)

    def _reiniciar_amp(self):
        self._anim_timer.stop()
        self._anim_frame = 0
        if self._anim_marcador is not None:
            self._anim_marcador.set_data([], [])
            self._canvas.draw_idle()

    def _cambiar_vel_amp(self, idx):
        self._anim_idx_vel = idx
        self._lbl_vel_amp.setText(f"×{self._VELOCIDADES_AMP[idx]}")

    def _sig_frame_amp(self):
        paso = self._VELOCIDADES_AMP[self._anim_idx_vel]
        self._anim_frame = min(self._anim_frame + paso, self._anim_n - 1)
        if self._anim_marcador is not None:
            self._anim_marcador.set_data(
                [self._anim_x[self._anim_frame]],
                [self._anim_y[self._anim_frame]],
            )
        self._canvas.draw_idle()
        if self._anim_frame >= self._anim_n - 1:
            self._anim_timer.stop()

    def closeEvent(self, event):
        if hasattr(self, '_anim_timer'):
            self._anim_timer.stop()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fig.tight_layout()
        self._canvas.draw_idle()


# ==============================================================================
# Diálogo de selección de panel para exportación
# ==============================================================================

class DialogExportarPanel(QDialog):
    """Pregunta al usuario qué panel (o todos) exportar."""

    _OPCIONES = [
        ("todos",  "Todos los paneles"),
        ("traj",   "Trayectoria orbital"),
        ("pot",    "Potencial efectivo"),
        ("radial", "Evolución radial"),
        ("dilat",  "Dilatación temporal"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar exportación")
        self.setFixedSize(320, 270)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("¿Qué desea exportar?"))

        self.lista = QListWidget()
        for _, nombre in self._OPCIONES:
            self.lista.addItem(nombre)
        self.lista.setCurrentRow(0)
        layout.addWidget(self.lista)

        btns = QHBoxLayout()
        btn_ok = QPushButton("Exportar")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def opcion_elegida(self):
        row = self.lista.currentRow()
        if 0 <= row < len(self._OPCIONES):
            return self._OPCIONES[row][0]
        return "todos"


# ==============================================================================
# Tab 4: Referencia estática de dilatación temporal
# ==============================================================================

class TabDilatacionEstatica(QWidget):
    """
    Pestaña con la curva estática dτ/dt = √(1-2M/r) frente a r/M.
    Disponible desde el arranque, sin necesidad de ninguna simulación.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.figura = Figure(figsize=(10, 7), tight_layout=True)
        self.figura.patch.set_facecolor(COLORES['fondo_fig'])
        self._ax = self.figura.add_subplot(1, 1, 1)

        self._canvas = FigureCanvas(self.figura)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas)

        graficar_dilatacion_estatica_general(self._ax)
        self._canvas.draw()

    def get_figura(self):
        return self.figura


# ==============================================================================
# Ventana principal
# ==============================================================================

class VentanaPrincipal(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Geodésicas en Schwarzschild — TFG de Física (G=c=M=1)"
        )
        self.resize(1300, 800)
        self._aplicar_estilo_qt()

        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout_central = QVBoxLayout(central)
        layout_central.setContentsMargins(4, 4, 4, 4)
        layout_central.setSpacing(0)

        # Barra superior con título y botón exportar
        barra_top = QHBoxLayout()
        titulo = QLabel("Explorador de Geodésicas de Schwarzschild")
        titulo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        barra_top.addWidget(titulo)
        barra_top.addStretch()

        self.btn_exportar = QPushButton("💾 Exportar para TFG")
        self.btn_exportar.setMinimumHeight(36)
        self.btn_exportar.setStyleSheet(
            "QPushButton { background-color: #226622; color: white; "
            "font-weight: bold; border-radius: 5px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #338833; }"
        )
        self.btn_exportar.clicked.connect(self._exportar)
        barra_top.addWidget(self.btn_exportar)

        self.btn_exportar_separado = QPushButton("🖼 Exportar paneles por separado")
        self.btn_exportar_separado.setMinimumHeight(36)
        self.btn_exportar_separado.setStyleSheet(
            "QPushButton { background-color: #224466; color: white; "
            "font-weight: bold; border-radius: 5px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #335577; }"
        )
        self.btn_exportar_separado.clicked.connect(self._exportar_paneles_por_separado)
        barra_top.addWidget(self.btn_exportar_separado)
        layout_central.addLayout(barra_top)

        # Pestañas
        self.tabs = QTabWidget()
        self.tab_simulador      = TabSimulador()
        self.tab_explorador     = TabExplorador()
        self.tab_comparador     = TabComparador()
        self.tab_dilat_estatica = TabDilatacionEstatica()

        self.tabs.addTab(self.tab_simulador,      "Simulador principal")
        self.tabs.addTab(self.tab_explorador,     "Explorador de potencial")
        self.tabs.addTab(self.tab_comparador,     "Comparador de trayectorias")
        self.tabs.addTab(self.tab_dilat_estatica, "Referencia estática")

        layout_central.addWidget(self.tabs)

        # Barra de estado
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Listo. Seleccione un preset o configure los parámetros manualmente."
        )

    def _aplicar_estilo_qt(self):
        """Estilo oscuro para la interfaz Qt."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1A1A2E;
                color: #E0E0E0;
            }
            QGroupBox {
                border: 1px solid #3A3A5A;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #AAAAFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background-color: #0D1117;
                color: #E0E0E0;
                border: 1px solid #3A3A5A;
                border-radius: 3px;
                padding: 2px 4px;
            }
            QPushButton {
                background-color: #2A2A4A;
                color: #E0E0E0;
                border: 1px solid #4A4A6A;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #3A3A6A; }
            QPushButton:pressed { background-color: #1A1A3A; }
            QPushButton:disabled { background-color: #222233; color: #666666; }
            QTabWidget::pane {
                border: 1px solid #3A3A5A;
            }
            QTabBar::tab {
                background-color: #16213E;
                color: #AAAACC;
                padding: 6px 16px;
                border: 1px solid #3A3A5A;
            }
            QTabBar::tab:selected {
                background-color: #1A1A4A;
                color: #FFFFFF;
            }
            QScrollArea, QListWidget {
                background-color: #0D1117;
                border: 1px solid #3A3A5A;
            }
            QRadioButton, QCheckBox { color: #E0E0E0; }
            QLabel { color: #CCCCCC; }
            QStatusBar { color: #AAAAAA; background-color: #0D1117; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3A3A5A;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4488FF;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #4488FF; border-radius: 2px; }
        """)

    def _exportar(self):
        """
        Exporta la figura de la pestaña activa a JPG y PDF.

        Para el simulador principal (pestaña 0) pregunta primero qué exportar:
        todos los paneles o uno concreto. En cualquier caso guarda también un
        .txt con los metadatos de la simulación cuando hay resultado disponible.
        El nombre base lo elige el usuario; si lo deja vacío se genera uno
        automático. Nunca sobrescribe archivos existentes: añade _01, _02, etc.
        """
        idx = self.tabs.currentIndex()

        # Solo para el simulador principal: preguntar qué exportar
        if idx == 0:
            dlg_sel = DialogExportarPanel(self)
            if dlg_sel.exec() != QDialog.DialogCode.Accepted:
                return
            opcion = dlg_sel.opcion_elegida()
        else:
            opcion = "todos"

        # Elegir carpeta de destino
        directorio = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de exportación",
            os.path.expanduser("~")
        )
        if not directorio:
            return

        # Resultado actual (solo tab simulador) para auto-nombre y TXT
        resultado_actual = self.tab_simulador.get_resultado() if idx == 0 else None

        # Sugerencia de nombre automático
        sugerencia = _generar_nombre_auto(idx, resultado_actual)
        if idx == 3:
            sugerencia = 'dilatacion_temporal_estatica_general'

        # Pedir nombre base al usuario
        nombre_input, ok = QInputDialog.getText(
            self,
            "Nombre del archivo",
            f"Nombre base para los archivos exportados\n"
            f"(vacío = nombre automático: {sugerencia}):",
        )
        if not ok:
            return
        nombre_input = nombre_input.strip()
        nombre_base = _sanitizar_nombre(nombre_input) if nombre_input else sugerencia

        # Rutas libres de colisión
        ruta_jpg, ruta_pdf, ruta_txt = _resolver_sin_colision(directorio, nombre_base)

        if opcion == "todos":
            figuras_tab = [
                self.tab_simulador.get_figura(),
                self.tab_explorador.get_figura(),
                self.tab_comparador.get_figura(),
                self.tab_dilat_estatica.get_figura(),
            ]
            try:
                _guardar_jpg_pdf(figuras_tab[idx], ruta_jpg, ruta_pdf)
            except Exception as e:
                QMessageBox.critical(self, "Error de exportación",
                                     f"No se pudo guardar la imagen:\n{e}")
                return

            ruta_txt_guardada = None
            if idx == 0 and resultado_actual is not None:
                try:
                    _guardar_metadatos_txt(
                        resultado_actual, ruta_txt,
                        metodo=self.tab_simulador.controles.cmb_metodo.currentText(),
                        tolerancia=self.tab_simulador.controles.cmb_tol.currentText(),
                        lambda_max=self.tab_simulador.controles.spin_lam_max.value(),
                    )
                    ruta_txt_guardada = ruta_txt
                except Exception as e:
                    QMessageBox.warning(self, "Advertencia",
                                        f"No se pudieron guardar los metadatos:\n{e}")
            elif idx == 2:
                configs_comp    = self.tab_comparador.configs_tray
                resultados_comp = self.tab_comparador.get_resultados()
                if configs_comp and resultados_comp:
                    try:
                        _guardar_metadatos_comparador_txt(
                            configs_comp, resultados_comp, ruta_txt,
                            nombre=nombre_base,
                        )
                        ruta_txt_guardada = ruta_txt
                    except Exception as e:
                        QMessageBox.warning(self, "Advertencia",
                                            f"No se pudieron guardar los metadatos:\n{e}")

            archivos = f"• {ruta_jpg}\n• {ruta_pdf}"
            if ruta_txt_guardada:
                archivos += f"\n• {ruta_txt_guardada}"
            QMessageBox.information(self, "Exportación completada",
                                    f"Archivos guardados:\n\n{archivos}")
            self.status.showMessage(
                f"Exportado: {ruta_jpg}, {ruta_pdf}" +
                (f" y {ruta_txt_guardada}" if ruta_txt_guardada else "")
            )

        else:
            # Exportar un panel concreto del simulador principal
            resultado = self.tab_simulador.get_resultado()
            if resultado is None:
                QMessageBox.warning(self, "Sin datos",
                                    "No hay ninguna simulación disponible para exportar.")
                return

            M_solar = self.tab_simulador.controles.grupo_masa.get_masa_solar()
            E       = resultado['E']
            L       = resultado['L']
            epsilon = resultado['epsilon']

            panel_map = {
                "traj":   0,
                "pot":    1,
                "radial": 2,
                "dilat":  3,
            }
            panel_idx = panel_map[opcion]

            fig_exp = Figure(figsize=(12, 10), tight_layout=True)
            fig_exp.patch.set_facecolor(COLORES['fondo_fig'])
            _canvas_exp = FigureCanvas(fig_exp)  # necesario para renderizado
            ax_exp = fig_exp.add_subplot(1, 1, 1)

            if panel_idx == 0:
                graficar_trayectoria(ax_exp, resultado, M_solar=M_solar)
            elif panel_idx == 1:
                graficar_potencial(ax_exp, L, epsilon, E)
            elif panel_idx == 2:
                graficar_evolucion_radial(ax_exp, resultado)
            else:
                graficar_dilatacion_temporal(ax_exp, resultado)

            try:
                _guardar_jpg_pdf(fig_exp, ruta_jpg, ruta_pdf)
            except Exception as e:
                QMessageBox.critical(self, "Error de exportación",
                                     f"No se pudo guardar la imagen:\n{e}")
                return

            ruta_txt_guardada = None
            try:
                _guardar_metadatos_txt(
                    resultado, ruta_txt,
                    metodo=self.tab_simulador.controles.cmb_metodo.currentText(),
                    tolerancia=self.tab_simulador.controles.cmb_tol.currentText(),
                    lambda_max=self.tab_simulador.controles.spin_lam_max.value(),
                )
                ruta_txt_guardada = ruta_txt
            except Exception as e:
                QMessageBox.warning(self, "Advertencia",
                                    f"No se pudieron guardar los metadatos:\n{e}")

            archivos = f"• {ruta_jpg}\n• {ruta_pdf}"
            if ruta_txt_guardada:
                archivos += f"\n• {ruta_txt_guardada}"
            QMessageBox.information(self, "Exportación completada",
                                    f"Archivos guardados:\n\n{archivos}")
            self.status.showMessage(
                f"Exportado: {ruta_jpg}, {ruta_pdf}" +
                (f" y {ruta_txt_guardada}" if ruta_txt_guardada else "")
            )

    def _exportar_paneles_por_separado(self):
        """
        Exporta cada panel del simulador principal como un JPG independiente (dpi=300).

        El prefijo del nombre de archivo se obtiene del preset activo;
        si no hay ninguno cargado, usa 'simulacion_actual'.
        Los paneles que fallen individualmente no interrumpen la exportación del resto.
        """
        resultado = self.tab_simulador.get_resultado()
        if resultado is None:
            QMessageBox.warning(
                self, "Sin datos",
                "Ejecute primero una simulación para poder exportar los paneles."
            )
            return

        directorio = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de destino",
            os.path.expanduser("~")
        )
        if not directorio:
            return

        preset_nombre = self.tab_simulador.controles.get_preset_activo()
        prefijo = _sanitizar_nombre(preset_nombre) if preset_nombre else "simulacion_actual"

        M_solar = self.tab_simulador.controles.grupo_masa.get_masa_solar()
        E       = resultado['E']
        L       = resultado['L']
        epsilon = resultado['epsilon']

        paneles = [
            ("trayectoria_orbital",
             lambda ax: graficar_trayectoria(ax, resultado, M_solar=M_solar)),
            ("potencial_efectivo",
             lambda ax: graficar_potencial(ax, L, epsilon, E)),
            ("evolucion_radial",
             lambda ax: graficar_evolucion_radial(ax, resultado)),
            ("dilatacion_temporal",
             lambda ax: graficar_dilatacion_temporal(ax, resultado)),
        ]

        exportados = []
        errores    = []

        for sufijo, dibujar_fn in paneles:
            nombre_archivo = f"{prefijo}_{sufijo}.jpg"
            ruta = os.path.join(directorio, nombre_archivo)
            try:
                fig_panel = Figure(figsize=(12, 10), tight_layout=True)
                fig_panel.patch.set_facecolor(COLORES['fondo_fig'])
                FigureCanvas(fig_panel)   # necesario para que matplotlib renderice
                ax_panel = fig_panel.add_subplot(1, 1, 1)
                dibujar_fn(ax_panel)
                _guardar_solo_jpg(fig_panel, ruta, dpi=300)
                exportados.append(nombre_archivo)
            except Exception as e:
                errores.append(f"{nombre_archivo}: {e}")

        n_ok    = len(exportados)
        n_total = len(paneles)
        msg = f"Se exportaron {n_ok} de {n_total} paneles en:\n{directorio}"
        if exportados:
            msg += "\n\nArchivos guardados:\n" + "\n".join(f"• {f}" for f in exportados)
        if errores:
            msg += "\n\nErrores:\n" + "\n".join(f"• {e}" for e in errores)

        QMessageBox.information(self, "Exportación completada", msg)
        self.status.showMessage(
            f"Paneles por separado: {n_ok}/{n_total} exportados en {directorio}"
        )
