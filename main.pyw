"""
main.pyw — Punto de entrada sin consola (modo usuario final).

Idéntico a main.py. La extensión .pyw hace que Windows lo ejecute
con pythonw.exe, suprimiendo la ventana de consola.

Uso:
    pythonw main.pyw          (sin consola — modo usuario final)
    python   main.py          (con consola — modo desarrollo)

Requisitos:
    pip install numpy scipy matplotlib PyQt6
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMessageBox
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
        msg = traceback.format_exc()
        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Error fatal — Schwarzschild Explorer", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
