"""Punto de entrada para ejecutar cada inciso del trabajo practico."""

from src.trayectoria_dron import ejecutar_extraccion
from src.ejecutar_ejercicio_3 import main as ejecutar_ejercicio_3
from src.renderizado_vuelo_real import renderizar_vuelo_real
from src.visualizacion_modelado import main as ejecutar_modelado


def seleccionar_item():
    """Mantiene el menu abierto hasta recibir un inciso valido."""
    print("\nSeleccione el item que desea ejecutar:")
    print("1 - Extraccion de la trayectoria")
    print("2 - Modelado y visualizacion de la trayectoria")
    print("3 - Renderizado del caso A y B, y luego su análisis")
    print("4 - Renderizado del vuelo con dinamica real")

    while True:
        opcion = input("Ingrese 1, 2, 3 o 4: ").strip()
        if opcion in {"1", "2", "3", "4"}:
            return opcion
        print("Opcion invalida. Ingrese 1, 2, 3 o 4.")


if __name__ == "__main__":
    item = seleccionar_item()
    if item == "1":
        ejecutar_extraccion()
    elif item == "2":
        ejecutar_modelado()
    elif item == "3":
        ejecutar_ejercicio_3()
    else:
        renderizar_vuelo_real()
