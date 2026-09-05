"""Menu principal de los incisos del trabajo practico."""

from src.trayectoria_dron import ejecutar_extraccion
from src.visualizacion_modelado import main as ejecutar_modelado


def seleccionar_item():
    """Solicita al usuario el inciso que desea ejecutar."""
    print("\nSeleccione el item que desea ejecutar:")
    print("1 - Extraccion de la trayectoria")
    print("2 - Modelado y visualizacion de la trayectoria")

    while True:
        opcion = input("Ingrese 1 o 2: ").strip()
        if opcion in {"1", "2"}:
            return opcion
        print("Opcion invalida. Ingrese 1 o 2.")


if __name__ == "__main__":
    item = seleccionar_item()
    if item == "1":
        ejecutar_extraccion()
    else:
        ejecutar_modelado()
