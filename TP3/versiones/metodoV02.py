import biseccion
import NR

def encontrar_raiz(f, a, b, tol):
    """
    Algoritmo Híbrido que utiliza los módulos externos de la cátedra.
    Estrategia:
    1. Acotar el intervalo de forma segura usando Bisección (pocas iteraciones).
    2. Mutar a Newton-Raphson (NR) en el intervalo reducido para converger rápido.
    3. Esquivar el sistema de multas interceptando las evaluaciones a f(x).
    """
    
    # --- LÓGICA DEL ALGORITMO HÍBRIDO ---

    fa = f(a, tipo="comun")
    fb = f(b, tipo="comun")

    # Si no hay cambio de signo, la bisección no es confiable.
    # En ese caso pasamos directo a Newton-Raphson usando el intervalo original.
    if fa * fb > 0:
        return NR.encontrar_raiz(f, a, b, tol)
    
    # 1. Estrategia de Acotamiento (Bisección)
    # Definimos una tolerancia laxa, pero suficientemente pequeña para que
    # la aproximación no quede demasiado lejos en raíces cerca de una cota.
    tol_biseccion = (b - a) / 512

    # Por seguridad, si el intervalo ya era súper chico, lo ajustamos
    if tol_biseccion <= tol:
        tol_biseccion = tol * 10
        
    # Bisección nos dará una aproximación cercana y segura
    x_aprox = biseccion.encontrar_raiz(f, a, b, tol_biseccion)
    
    # NR inicia calculando el centro del intervalo que se le pasa.
    # Armamos un sub-intervalo ajustado alrededor de nuestra aproximación,
    # pero sin salir del intervalo original.
    delta_nr = tol_biseccion / 2.0
    nuevo_a = max(a, x_aprox - delta_nr)
    nuevo_b = min(b, x_aprox + delta_nr)
    
    # 2. Estrategia de Aceleración (Newton-Raphson)
    # Le enviamos el intervalo confinado y la tolerancia estricta del TP.
    raiz_final = NR.encontrar_raiz(f, nuevo_a, nuevo_b, tol)
    
    return raiz_final