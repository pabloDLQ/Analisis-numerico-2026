from src.convergencia import (
    calcular_errores_y_ordenes,
    graficar_curvas_error,
    mostrar_resultados_consola,
)
from src.simulacion import ejecutar_simulacion_1b


def main():
    """Presenta el menú y ejecuta la opción seleccionada por el usuario."""
    print("¿Qué desea hacer?")
    print("1. Gráfica de evolución del sistema")
    print("2. Resultados numéricos de convergencia")
    print("3. Gráficas de curvas de error")
    opcion = input("Ingrese una opción: ")

    if opcion == "1":
        print("Calculando integración y generando gráfica de evolución...")
        ejecutar_simulacion_1b()
        print("Operación finalizada. Gráfico guardado en la carpeta 'graficos/'.")

    elif opcion == "2":
        print("Calculando errores y evaluando convergencia...")
        _, _, ordenes, _, _ = calcular_errores_y_ordenes()
        mostrar_resultados_consola(ordenes)
        print("Operación finalizada. Resultados mostrados por pantalla.")

    elif opcion == "3":
        dt_lista, errores, ordenes, t1, t2 = calcular_errores_y_ordenes()
        graficar_curvas_error(dt_lista, errores, ordenes, t1, t2)

    else:
        print("Opción no válida.")


if __name__ == "__main__":
    main()
