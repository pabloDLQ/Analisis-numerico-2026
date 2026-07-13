# =========================================================================
# ARCHIVO: grupo_01.py (NEWTON-RAPHSON PURO - CON DECLARACIÓN DE LLAMADAS)
# =========================================================================

def encontrar_raiz(f, a, b, tol):
    """
    Algoritmo de Newton-Raphson Puro con aproximación de derivada por 
    diferencias finitas hacia adelante.
    Declara explícitamente a la caja negra el tipo de llamada para el peaje.
    """
    # Tomamos el punto medio del intervalo [a, b] como aproximación inicial
    x = (a + b) / 2.0
    dx = 1e-6 
    max_iter = 150
    
    for _ in range(max_iter):
        # LLAMADA 1: Exploración común (Por defecto o explícito como "comun")
        fx = f(x, tipo="comun")
        
        if fx == 0.0:
            return x
            
        # --- CÁLCULO DE LA DERIVADA NUMÉRICA HACIA ADELANTE ---
        # LLAMADA 2: Le avisamos al evaluador que esto es estrictamente una derivada.
        # El peaje procesará el subsidio de 0.2 de forma directa y honesta.
        fx_derecha = f(x + dx, tipo="derivada")
        
        dfx = (fx_derecha - fx) / dx
        
        # Salvaguarda crítica: Evitar división por cero en zonas planas
        if abs(dfx) < 1e-11:
            x = (x + b) / 2.0 if fx > 0 else (a + x) / 2.0
            continue
            
        # --- PASO DE NEWTON-RAPHSON ---
        x_siguiente = x - fx / dfx
        
        # Guardián de Confinamiento de seguridad
        if not (a < x_siguiente < b):
            x_siguiente = (a + b) / 2.0
        
        # Criterio de parada en el eje X
        if abs(x_siguiente - x) <= tol:
            return x_siguiente
            
        x = x_siguiente
        
    return x