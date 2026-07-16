import math

def encontrar_raiz(f, a, b, tol):
    """
    Argumentos de entrada:
    f : Función a evaluar. Se debe invocar especificando el "tipo" de llamada:
    - Para evaluar la función de forma estándar: f(x, tipo="comun")
    - Para evaluar puntos destinados al cálculo de derivadas: f(x, tipo="derivada")
    a, b: Flotantes. Extremos del intervalo inicial (b > a).
    tol : Flotante>0. Tolerancia de convergencia.
    El algoritmo DEBE detenerse estrictamente cuando | x_k - x_{k-1} | <= tol.
    Para la evaluación, el equipo docente utilizará: tol = 1e-12
    Retorno:
    Devuelve únicamente un número flotante con la estimación de la raíz.
    """
    # Ejemplo de llamadas válidas para la telemetría del peaje (sistema de benchmark):
    # fx = f(x, tipo="comun")
    # fx_der = f(x + dx, tipo="derivada")

    max_iter = 150

    x_anterior = a
    f_anterior = f(a, tipo="comun")
    if f_anterior == 0.0:
        return x_anterior

    x_actual = b
    f_actual = f(b, tipo="comun")
    if f_actual == 0.0:
        return x_actual

    # Mejor aproximación (para devolver si se agotan las iteraciones)
    mejor_x = x_actual
    mejor_f = abs(f_actual)

    for _ in range(max_iter):
        denominador = f_actual - f_anterior

        # --- SALVAGUARDA 1: Denominador cero (zona plana) ---
        if denominador == 0.0:
            # Exactamente como NR.py: usamos extremos fijos del intervalo original
            if f_actual > 0:
                x_siguiente = (x_actual + b) / 2.0
            else:
                x_siguiente = (a + x_actual) / 2.0
        else:
            # Fórmula estándar de la secante
            x_siguiente = x_actual - f_actual * (x_actual - x_anterior) / denominador

        # --- SALVAGUARDA 2: Confinamiento al intervalo original ---
        if not (a <= x_siguiente <= b):
            # Reubicamos en el centro del intervalo original (igual que NR.py)
            x_siguiente = (a + b) / 2.0

        # Criterio de parada estricto (|x_k - x_{k-1}| <= tol)
        if abs(x_siguiente - x_actual) <= tol:
            return x_siguiente

        # Evaluación de la función en el nuevo punto
        f_siguiente = f(x_siguiente, tipo="comun")
        if f_siguiente == 0.0:
            return x_siguiente

        # Actualizar mejor aproximación (solo para el caso de no convergencia)
        if abs(f_siguiente) < mejor_f:
            mejor_x = x_siguiente
            mejor_f = abs(f_siguiente)

        # Actualización de la secante (descartar el más antiguo)
        x_anterior, f_anterior = x_actual, f_actual
        x_actual, f_actual = x_siguiente, f_siguiente

    # Si se agotan las iteraciones, devolvemos la mejor aproximación encontrada
    return mejor_x