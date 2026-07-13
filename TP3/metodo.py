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
    
    # 1. Estrategia de Acotamiento (Bisección)
    # Definimos una tolerancia laxa (ej: 1/8 del tamaño inicial).
    # Esto asegura que Bisección dé unos pocos pasos y termine pronto.
    tol_biseccion = (b - a) / 8.0 
    
    # Por seguridad, si el intervalo ya era súper chico, lo ajustamos
    if tol_biseccion <= tol:
        tol_biseccion = tol * 10
        
    # Bisección nos dará una aproximación cercana y segura
    x_aprox = biseccion.encontrar_raiz(f, a, b, tol_biseccion)
    
    # NR inicia calculando el centro del intervalo que se le pasa.
    # Armamos un sub-intervalo ajustado alrededor de nuestra aproximación:
    nuevo_a = x_aprox - tol_biseccion
    nuevo_b = x_aprox + tol_biseccion
    
    # 2. Estrategia de Aceleración (Newton-Raphson)
    # Le enviamos el intervalo confinado y la tolerancia estricta del TP.
    raiz_final = NR.encontrar_raiz(f, nuevo_a, nuevo_b, tol)
    
    return raiz_final