# Se importa Path para construir rutas independientes del sistema operativo.
from pathlib import Path

# Se importan las funciones que ejecutan cada ejercicio del trabajo práctico.
from src.analisis_error_pluvial import ejecutar_analisis_error
from src.calcular_areas import analizar_techos, validar_area_rectangulo
from src.calculo_irradiancia_diaria import ejecutar_calculo_irradiancia
from src.calculo_fotovoltaico import ejecutar_calculo_fotovoltaico
from src.calculo_pluvial import ejecutar_calculo_pluvial
from src.generar_imagen_techos import dibujar_techos_sobre_mapa
from src.leer_datos import leer_techos


# Se toma como carpeta base la ubicación de este archivo.
RUTA_BASE = Path(__file__).parent
# Se definen las rutas de los datos de entrada y de la imagen de salida.
RUTA_DATOS = RUTA_BASE / "data" / "techos_campus_utn.npz"
RUTA_IMAGEN = RUTA_BASE / "data" / "mapa_satelital_completo.jpg"
RUTA_SALIDA = RUTA_BASE / "resultados" / "ejercicio1" / "techos_sobre_mapa.png"


def calcular_techos() -> None:
    """Lee los techos, calcula sus áreas y genera la imagen."""
    # Se cargan los contornos de los techos almacenados en el archivo NPZ.
    techos = leer_techos(RUTA_DATOS)
    print(f"Archivo: {RUTA_DATOS}")
    # Se comprueba que la fórmula de área funcione antes de procesar los datos.
    validar_area_rectangulo()
    # Se calculan las áreas reales y se dibujan sobre la imagen satelital.
    areas_m2 = analizar_techos(techos)
    dibujar_techos_sobre_mapa(techos, areas_m2, RUTA_IMAGEN, RUTA_SALIDA)


def main() -> None:
    # Se muestran las operaciones disponibles para el usuario.
    print("¿Qué desea hacer?")
    print("1. Calcular techos")
    print("2. Calcular radiación solar")
    print("3. Cálculo de generación fotovoltaica")
    print("4. Modelado de descarga pluvial")
    # Se lee la opción como texto para compararla con las alternativas del menú.
    opcion = input("Ingrese una opción: ")

    # Se ejecuta únicamente el análisis correspondiente a la opción elegida.
    if opcion == "1":
        calcular_techos()
    elif opcion == "2":
        ejecutar_calculo_irradiancia()
    elif opcion == "3":
        ejecutar_calculo_fotovoltaico()
    elif opcion == "4":
        ejecutar_calculo_pluvial()
        # El análisis de error usa los mismos cálculos del modelo pluvial.
        ejecutar_analisis_error()
    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
