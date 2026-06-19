"""
main.py — Punto de entrada de la aplicación.

Inicia la interfaz gráfica de la herramienta de exploración de geodésicas
en el espacio-tiempo de Schwarzschild.

Uso:
    python main.py

Requisitos:
    pip install numpy scipy matplotlib PyQt6
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui import VentanaPrincipal


def main():
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        app = QApplication(sys.argv)
        app.setApplicationName("Geodesicas de Schwarzschild")
        app.setOrganizationName("TFG Fisica UNIR")

        ventana = VentanaPrincipal()
        ventana.show()

        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        input("Pulsa ENTER para cerrar...")


if __name__ == "__main__":
    main()
