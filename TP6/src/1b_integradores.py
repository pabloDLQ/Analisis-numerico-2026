"""Integradores temporales explícitos para sistemas de ecuaciones diferenciales.

Este módulo implementa únicamente operaciones propias con NumPy; no utiliza
ningún solucionador automático de ecuaciones diferenciales.
"""

import numpy as np


def integrar_edo(funcion_derivada, t_span, X0, dt, metodo):
    """Integra ``dX/dt = funcion_derivada(t, X)`` en un intervalo dado.

    El último paso se ajusta si es necesario para que ``t_out`` termine
    exactamente en el extremo derecho de ``t_span``.
    """

    # Se convierten los extremos a escalares reales para evitar aritmética entera.
    t_inicial, t_final = map(float, t_span)
    # Se convierte el estado inicial en un vector real independiente del original.
    estado_inicial = np.asarray(X0, dtype=float).copy()
    # Se exige un vector unidimensional, porque el sistema es de primer orden vectorial.
    if estado_inicial.ndim != 1:
        raise ValueError("X0 debe ser un vector unidimensional.")
    # Se verifica que el intervalo tenga orientación temporal válida.
    if t_final < t_inicial:
        raise ValueError("El extremo final debe ser mayor o igual que el inicial.")
    # Se verifica que el paso temporal sea finito y estrictamente positivo.
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt debe ser un número finito estrictamente positivo.")
    # Se normaliza el nombre del método para aceptar mayúsculas y minúsculas.
    nombre_metodo = str(metodo).strip().lower()
    # Se rechazan anticipadamente los métodos que no están implementados.
    if nombre_metodo not in ("euler", "rk4"):
        raise ValueError("metodo debe ser 'euler' o 'rk4'.")
    # Se construye una lista mutable para poder corregir el último paso.
    tiempos = [t_inicial]
    # Se construye una lista de estados, comenzando con una copia de X0.
    estados = [estado_inicial.copy()]
    # Se inicializa el tiempo y el estado que avanzarán durante la integración.
    tiempo_actual = t_inicial
    estado_actual = estado_inicial.copy()

    # Se avanza mientras todavía no se alcanzó el extremo final.
    while tiempo_actual < t_final:
        # Se limita el paso para no sobrepasar t_final por redondeo o por dt.
        paso = min(float(dt), t_final - tiempo_actual)
        # Se calcula la pendiente en el punto actual para el método de Euler.
        k1 = np.asarray(funcion_derivada(tiempo_actual, estado_actual), dtype=float)
        # Se comprueba que la función derivada conserve la dimensión del estado.
        if k1.shape != estado_actual.shape:
            raise ValueError("La derivada debe tener la misma forma que X0.")
        # Se realiza el avance correspondiente al método explícito elegido.
        if nombre_metodo == "euler":
            # Euler usa la pendiente inicial constante durante todo el paso.
            estado_nuevo = estado_actual + paso * k1
        else:
            # RK4 evalúa cuatro pendientes para aproximar la pendiente media del paso.
            k2 = np.asarray(
                funcion_derivada(tiempo_actual + paso / 2.0, estado_actual + paso * k1 / 2.0),
                dtype=float,
            )
            # Se verifica la dimensión de la segunda pendiente antes de combinarla.
            if k2.shape != estado_actual.shape:
                raise ValueError("La derivada debe tener la misma forma que X0.")
            # La tercera pendiente usa el estado aproximado por k2 en el punto medio.
            k3 = np.asarray(
                funcion_derivada(tiempo_actual + paso / 2.0, estado_actual + paso * k2 / 2.0),
                dtype=float,
            )
            # Se verifica la dimensión de la tercera pendiente.
            if k3.shape != estado_actual.shape:
                raise ValueError("La derivada debe tener la misma forma que X0.")
            # La cuarta pendiente usa el estado aproximado por k3 en el extremo derecho.
            k4 = np.asarray(
                funcion_derivada(tiempo_actual + paso, estado_actual + paso * k3),
                dtype=float,
            )
            # Se verifica la dimensión de la cuarta pendiente.
            if k4.shape != estado_actual.shape:
                raise ValueError("La derivada debe tener la misma forma que X0.")
            # Se combina la regla 1-2-2-1 de RK4, de orden global cuatro.
            estado_nuevo = estado_actual + paso * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        # Se actualiza el tiempo acumulando exactamente el paso aplicado.
        tiempo_actual += paso
        # Se fuerza el valor final para evitar una pequeña discrepancia de redondeo.
        if np.isclose(tiempo_actual, t_final, rtol=0.0, atol=1e-14):
            tiempo_actual = t_final
        # Se guarda el nuevo estado sin aliasar la memoria del estado anterior.
        estado_actual = np.asarray(estado_nuevo, dtype=float)
        # Se agrega el tiempo alcanzado a la salida.
        tiempos.append(tiempo_actual)
        # Se agrega el estado alcanzado a la salida.
        estados.append(estado_actual.copy())

    # Se convierte la lista de tiempos en un arreglo NumPy unidimensional.
    t_out = np.asarray(tiempos, dtype=float)
    # Se convierte la lista de estados en una matriz de filas temporales.
    X_out = np.vstack(estados)
    # Se devuelven conjuntamente la grilla temporal y la solución aproximada.
    return t_out, X_out
