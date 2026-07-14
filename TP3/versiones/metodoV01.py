import biseccion
import NR

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
    
    # usamos una tolerancia amplia para bisección.
    tol_biseccion = (b - a) / 16.0 # son 16 porque son 2^4 iteraciones asi no hay penalizacion monotona

    # si el intervalo ya es chico, subimos un poco la tolerancia.
    if tol_biseccion <= tol:
        tol_biseccion = tol * 10

    # bisección da una primera aproximación.
    x_aprox = biseccion.encontrar_raiz(f, a, b, tol_biseccion)

    # armamos un intervalo chico alrededor de esa aproximación.
    nuevo_a = max(a, x_aprox - tol_biseccion)
    nuevo_b = min(b, x_aprox + tol_biseccion)

    # NR termina de refinar la raíz.
    raiz_estimada = NR.encontrar_raiz(f, nuevo_a, nuevo_b, tol)

    return raiz_estimada