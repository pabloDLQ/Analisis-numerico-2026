"""Sistema MOL acoplado entre concentración bidimensional y navegación."""

import numpy as np

from .operadores_espaciales import (
    aplicar_condiciones_contorno,
    calcular_adveccion_upwind,
    calcular_baricentro,
    calcular_difusion,
    D,
    UV,
    VV,
    KM,
    VM,
)

# Rapidez constante de navegación del dron, en m/s.
VELOCIDAD_DRON_EDP = 1.0
# Radio que define que el dron alcanzó el baricentro, en metros.
DISTANCIA_CAPTURA_EDP = 1.2


def crear_estado_inicial(malla):
    """Devuelve la concentración nula y el dron en (115, 115)."""
    # Se inicializa C=0 en los 131x131 nodos de la malla.
    concentracion_inicial = np.zeros(malla.Nx * malla.Ny)
    # Se agregan al final del vector las dos coordenadas del dron.
    posicion_inicial = np.array([115.0, 115.0])
    # La concatenación forma el estado único requerido por el integrador.
    return np.concatenate((concentracion_inicial, posicion_inicial))


def derivadas_sistema_edp(t, estado, malla):
    """Calcula las derivadas MOL de la concentración y del dron."""
    # Se convierte el estado a un arreglo real para realizar operaciones NumPy.
    vector = np.asarray(estado, dtype=float)
    # Las primeras entradas representan C y las dos últimas la posición del dron.
    tam_malla = malla.Nx * malla.Ny
    if vector.shape != (tam_malla + 2,):
        raise ValueError("El estado EDP tiene una dimensión inesperada.")
    C = aplicar_condiciones_contorno(
        vector[:tam_malla].reshape(malla.Ny, malla.Nx), t, malla
    )
    # Se desempaquetan las coordenadas actuales del dron.
    xd, yd = vector[-2:]
    # Se evalúan difusión, advección y reacción sobre el campo corregido.
    difusion = D * calcular_difusion(C, malla.dx, malla.dy)
    adveccion = calcular_adveccion_upwind(C, malla.dx, malla.dy, UV, VV)
    reaccion = -VM * C / (KM + np.maximum(C, 0.0))
    # La EDP combina difusión, advección con signo negativo y consumo reactivo.
    derivada_C = difusion - adveccion + reaccion
    _, xc, yc = calcular_baricentro(C, malla.X, malla.Y, malla.pesos_simpson)
    vector_objetivo = np.array([xc - xd, yc - yd])
    distancia = float(np.linalg.norm(vector_objetivo))
    if distancia <= np.finfo(float).eps:
        # Si el objetivo coincide exactamente con el dron, no se normaliza un vector nulo.
        derivada_dron = np.zeros(2)
    else:
        # El dron se desplaza a velocidad constante en dirección al baricentro.
        derivada_dron = VELOCIDAD_DRON_EDP * vector_objetivo / distancia
    # Las condiciones se imponen después del cálculo para que los bordes permanezcan fijos.
    derivada_C[0, :] = 0.0
    derivada_C[:, 0] = 0.0
    derivada_C[-1, :] = 0.0
    derivada_C[:, -1] = 0.0
    return np.concatenate((derivada_C.ravel(), derivada_dron))


def proyectar_contorno(t, estado, malla):
    """Proyecta las cuatro fronteras sobre un estado integrado."""
    # Se copia el vector para que la proyección no altere historiales previos.
    resultado = np.asarray(estado, dtype=float).copy()
    # Se reconstruye C, se proyectan sus fronteras y se vuelve a aplanar.
    C = aplicar_condiciones_contorno(
        resultado[:-2].reshape(malla.Ny, malla.Nx), t, malla
    )
    resultado[:-2] = C.ravel()
    return resultado


def distancia_captura(t, estado, malla):
    """Indica si el dron alcanzó el baricentro de la concentración."""
    # Se aplican las mismas fronteras usadas por la dinámica antes de medir distancia.
    C = aplicar_condiciones_contorno(
        estado[:-2].reshape(malla.Ny, malla.Nx), t, malla
    )
    # Se calcula el baricentro y se compara con la posición del dron.
    _, xc, yc = calcular_baricentro(C, malla.X, malla.Y, malla.pesos_simpson)
    return float(np.linalg.norm(np.array([xc, yc]) - estado[-2:])) <= DISTANCIA_CAPTURA_EDP
