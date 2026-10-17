from pathlib import Path

from src.calcular_areas import analizar_techos, validar_area_rectangulo
from src.calculo_irradiancia_diaria import ejecutar_calculo_irradiancia
from src.calculo_fotovoltaico import ejecutar_calculo_fotovoltaico
from src.generar_imagen_techos import dibujar_techos_sobre_mapa
from src.leer_datos import leer_techos


RUTA_BASE = Path(__file__).parent
RUTA_DATOS = RUTA_BASE / "data" / "techos_campus_utn.npz"
RUTA_IMAGEN = RUTA_BASE / "data" / "mapa_satelital_completo.jpg"
RUTA_SALIDA = RUTA_BASE / "resultados" / "ejercicio1" / "techos_sobre_mapa.png"


def calcular_techos() -> None:
    """Lee los techos, calcula sus áreas y genera la imagen."""
    techos = leer_techos(RUTA_DATOS)
    print(f"Archivo: {RUTA_DATOS}")
    validar_area_rectangulo()
    areas_m2 = analizar_techos(techos)
    dibujar_techos_sobre_mapa(techos, areas_m2, RUTA_IMAGEN, RUTA_SALIDA)


def main() -> None:
    print("¿Qué desea hacer?")
    print("1. Calcular techos")
    print("2. Calcular radiación solar")
    print("3. Cálculo de generación fotovoltaica")
    opcion = input("Ingrese una opción: ")

    if opcion == "1":
        calcular_techos()
    elif opcion == "2":
        ejecutar_calculo_irradiancia()
    elif opcion == "3":
        ejecutar_calculo_fotovoltaico()
    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
