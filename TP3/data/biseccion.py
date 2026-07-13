# =========================================================================
# ARCHIVO: grupo_01.py (BISECCIÓN PURA Y ESTRICTA - BASELINE DE CÁTEDRA)
# =========================================================================

def encontrar_raiz(f, a, b, tol):
    """
    Algoritmo de Bisección Estándar.
    Depende exclusivamente del Teorema de Bolzano (cambio de signo).
    """
    fa = f(a, tipo="comun")
    fb = f(b, tipo="comun")
    
    # Salvaguarda inmediata de extremos
    if fa == 0.0: return a
    if fb == 0.0: return b
    
    # Control estricto de Bolzano al inicio
    if fa * fb > 0:
        raise ValueError("El intervalo inicial no encierra una raíz (mismo signo).")
        
    # Bucle principal controlado puramente por el intervalo en X
    while (b - a) > tol:
        c = (a + b) / 2.0
        fc = f(c, tipo="comun")  # Suma 1 llamado por vuelta
        
        # Salvaguarda contra el cero flotante (underflow de la máquina)
        if fc == 0.0:
            return c
            
        # Reducción de intervalo por Teorema de Bolzano
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return (a + b) / 2.0