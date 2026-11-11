import importlib.util
from pathlib import Path

DIRECTORIO_RAIZ = Path(__file__).resolve().parent


def cargar_funcion(ruta, nombre_modulo, nombre_funcion):
    """Carga dinámicamente una función desde un archivo específico."""
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo en {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return getattr(modulo, nombre_funcion)


RUTA_SIMULACION_1B = DIRECTORIO_RAIZ / "src" / "1b_simulacion.py"
RUTA_CONVERGENCIA_1C = DIRECTORIO_RAIZ / "src" / "1c_convergencia.py"


def main():
    """Presenta el menú y ejecuta la opción seleccionada por el usuario."""
    print("¿Qué desea hacer?")
    print("1. Gráfica de evolución del sistema")
    print("2. Resultados numéricos de convergencia")
    print("3. Gráficas de curvas de error")
    opcion = input("Ingrese una opción: ")

    if opcion == "1":
        print("Calculando integración y generando gráfica de evolución...")
        ejecutar_simulacion_1b = cargar_funcion(RUTA_SIMULACION_1B, "simulacion_1b", "ejecutar_simulacion_1b")
        ejecutar_simulacion_1b()
        print("Operación finalizada. Gráfico guardado en la carpeta 'graficos/'.")

    elif opcion == "2":
        print("Calculando errores y evaluando convergencia...")
        try:
            calcular_errores_y_ordenes = cargar_funcion(
                RUTA_CONVERGENCIA_1C,
                "convergencia_1c",
                "calcular_errores_y_ordenes",
            )
            mostrar_resultados_consola = cargar_funcion(
                RUTA_CONVERGENCIA_1C,
                "convergencia_1c",
                "mostrar_resultados_consola",
            )
            _, _, ordenes, _, _ = calcular_errores_y_ordenes()
            mostrar_resultados_consola(ordenes)
            print("Operación finalizada. Resultados mostrados por pantalla.")
        except FileNotFoundError:
            print("Error: El archivo 1c_convergencia.py aún no ha sido creado o adaptado.")

    elif opcion == "3":
        try:
            calcular_errores_y_ordenes = cargar_funcion(
                RUTA_CONVERGENCIA_1C,
                "convergencia_1c",
                "calcular_errores_y_ordenes",
            )
            graficar_curvas_error = cargar_funcion(
                RUTA_CONVERGENCIA_1C,
                "convergencia_1c",
                "graficar_curvas_error",
            )
            dt_lista, errores, ordenes, t1, t2 = calcular_errores_y_ordenes()
            graficar_curvas_error(dt_lista, errores, ordenes, t1, t2)
        except FileNotFoundError:
            print("Error: El archivo 1c_convergencia.py aún no ha sido creado o adaptado.")

    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
