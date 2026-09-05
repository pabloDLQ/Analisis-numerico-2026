"""Menu principal de los incisos del trabajo practico."""

from src.trayectoria_dron import ejecutar_extraccion
from src.renderizado_caso_A import renderizar_caso_A
from src.renderizado_caso_B import renderizar_caso_B
from src.visualizacion_modelado import main as ejecutar_modelado


def seleccionar_item():
    """Solicita al usuario el inciso que desea ejecutar."""
    print("\nSeleccione el item que desea ejecutar:")
    print("1 - Extraccion de la trayectoria")
    print("2 - Modelado y visualizacion de la trayectoria")
    print("3 - Renderizado del caso A y B, y luego su análisis")

    while True:
        opcion = input("Ingrese 1, 2 o 3: ").strip()
        if opcion in {"1", "2", "3"}:
            return opcion
        print("Opcion invalida. Ingrese 1, 2 o 3.")


if __name__ == "__main__":
    item = seleccionar_item()
    if item == "1":
        ejecutar_extraccion()
    elif item == "2":
        ejecutar_modelado()
    else:
        renderizar_caso_A()
        renderizar_caso_B()
