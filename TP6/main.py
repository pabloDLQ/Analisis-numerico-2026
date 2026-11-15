from src.convergencia import (
    calcular_errores_y_ordenes,
    graficar_curvas_error,
    mostrar_resultados_consola,
)
from src.simulacion import ejecutar_simulacion_1b
from src.convergencia_edp import ejecutar_convergencia_edp
from src.simulacion_edp import ejecutar_simulacion_2a


def main():
    """Presenta el menú y ejecuta la opción seleccionada por el usuario."""

    print("¿Qué item quiere ejecutar?")
    print("1. Ítem 1: modelo EDO")
    print("2. Ítem 2: modelo EDP")
    opcion1 = input("Ingrese una opción: ")

    if opcion1 == "1":
        print("¿Qué desea hacer?")
        print("1. Gráfica de evolución del sistema")
        print("2. Resultados numéricos de convergencia")
        print("3. Gráficas de curvas de error")
        print("4. Ejecutar Ítem 1 completo")
        opcion2 = input("Ingrese una opción: ")

        if opcion2 == "1":
            print("Calculando integración y generando gráfica de evolución...")
            ejecutar_simulacion_1b()
            print("Operación finalizada. Gráfico guardado en 'resultados/item1/'.")

        elif opcion2 == "2":
            print("Calculando errores y evaluando convergencia...")
            _, _, ordenes, _, _ = calcular_errores_y_ordenes()
            mostrar_resultados_consola(ordenes)
            print("Operación finalizada. Resultados mostrados por pantalla.")

        elif opcion2 == "3":
            dt_lista, errores, ordenes, t1, t2 = calcular_errores_y_ordenes()
            graficar_curvas_error(dt_lista, errores, ordenes, t1, t2)

        elif opcion2 == "4":
            print("Ejecutando Ítem 1 completo...")
            ejecutar_simulacion_1b()
            dt_lista, errores, ordenes, t1, t2 = calcular_errores_y_ordenes()
            mostrar_resultados_consola(ordenes)
            graficar_curvas_error(dt_lista, errores, ordenes, t1, t2)

        else:
            print("Opción no válida.")
    elif opcion1 == "2":
        print("¿Qué desea hacer?")
        print("1. Simulación y captura (Ítem 2.a)")
        print("2. Convergencia temporal (Ítem 2.b)")
        print("3. Ejecutar Ítem 2 completo")
        opcion2 = input("Ingrese una opción: ")
        if opcion2 == "1":
            ejecutar_simulacion_2a()
        elif opcion2 == "2":
            ejecutar_convergencia_edp()
        elif opcion2 == "3":
            ejecutar_simulacion_2a()
            ejecutar_convergencia_edp()
        else:
            print("Opción no válida.")
        if opcion2 in ("1", "2", "3"):
            print("Las resoluciones del Ítem 2 quedaron guardadas en 'resultados/item2/'.")
    else:
        print("Opción no válida.")



if __name__ == "__main__":
    main()
