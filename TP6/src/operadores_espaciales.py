"""Malla, operadores espaciales y cuadratura del modelo de advección-difusión."""

from dataclasses import dataclass

import numpy as np


# Concentración máxima impuesta en las fronteras de entrada, en g/m³.
C_PICO = 105.0
# Desvío estándar de los perfiles gaussianos de entrada, en metros.
SIGMA_IN = 6.0
# Instante en que se apaga la fuente de las fronteras de entrada, en segundos.
T_INYECCION = 50.0
# Coeficiente de difusión molecular, en m²/s.
D = 0.5
# Componente horizontal constante de la velocidad del viento, en m/s.
UV = 0.1
# Componente vertical constante de la velocidad del viento, en m/s.
VV = 0.8
# Constante de Michaelis-Menten de la reacción, en g/m³.
KM = 20.0
# Velocidad máxima de consumo químico, en g/(s·m³).
VM = 1.0


@dataclass(frozen=True)
class MallaEDP:
    """Agrupa la malla regular y los pesos de integración del Ítem 2."""

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    pesos_simpson: np.ndarray
    dx: float
    dy: float
    Nx: int
    Ny: int
    dt_max: float


def crear_malla():
    """Construye la malla de 131 por 131 nodos y sus pesos de Simpson."""
    # Se generan los nodos de ambos ejes, incluidos los dos extremos del dominio.
    x = np.linspace(0.0, 130.0, 131)
    y = np.linspace(0.0, 130.0, 131)
    # Se forma una matriz de coordenadas para evaluar campos sobre la grilla.
    X, Y = np.meshgrid(x, y)
    # La separación es uniforme y se obtiene de los dos primeros nodos.
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # Se inicializan los pesos unidimensionales de Simpson compuesta.
    pesos_x = np.ones(131)
    pesos_y = np.ones(131)
    # Los nodos interiores impares reciben peso 4 y los pares peso 2.
    pesos_x[1:-1:2] = 4.0
    pesos_x[2:-1:2] = 2.0
    pesos_y[1:-1:2] = 4.0
    pesos_y[2:-1:2] = 2.0
    # El producto exterior construye los pesos bidimensionales de Simpson.
    pesos = np.outer(pesos_y, pesos_x) * dx * dy / 9.0
    # Se calcula la cota explícita combinando advección, difusión y reacción.
    dt_max = 1.0 / (UV / dx + VV / dy + 2.0 * D * (dx ** -2 + dy ** -2) + VM / KM)
    # Se agrupan todos los objetos para que los restantes módulos compartan la malla.
    return MallaEDP(x, y, X, Y, pesos, dx, dy, 131, 131, dt_max)


def aplicar_condiciones_contorno(C, t, malla):
    """Impone entradas de Dirichlet y salidas de Neumann homogéneas."""
    # Se copia el campo para no modificar accidentalmente el estado del integrador.
    resultado = np.asarray(C, dtype=float).copy()
    # La matriz debe tener filas asociadas a y y columnas asociadas a x.
    if resultado.ndim != 2:
        raise ValueError("C debe ser una matriz bidimensional.")
    # Si la fuente está encendida, se evalúan los perfiles sobre las coordenadas físicas.
    if t <= T_INYECCION:
        # La columna 0 representa x=0 y varía a lo largo de malla.y.
        entrada_x = C_PICO * np.exp(-(malla.y ** 2) / (2.0 * SIGMA_IN ** 2))
        # La fila 0 representa y=0 y varía a lo largo de malla.x.
        entrada_y = C_PICO * np.exp(-(malla.x ** 2) / (2.0 * SIGMA_IN ** 2))
        resultado[:, 0] = entrada_x
        resultado[0, :] = entrada_y
    else:
        # Después de la inyección, ambas fronteras de entrada quedan en cero.
        resultado[:, 0] = 0.0
        resultado[0, :] = 0.0
    # La última columna representa x=130: se copia su vecino para imponer dC/dx=0.
    resultado[:, -1] = resultado[:, -2]
    # La última fila representa y=130: se copia su vecino para imponer dC/dy=0.
    resultado[-1, :] = resultado[-2, :]
    return resultado


def calcular_difusion(C, dx, dy):
    """Calcula el laplaciano centrado de cinco puntos en nodos interiores."""
    # Se reserva un campo nulo; las fronteras no evolucionan mediante la EDP.
    salida = np.zeros_like(C, dtype=float)
    # 1:-1 selecciona nodos interiores, 2: sus vecinos derechos/superiores,
    # y :-2 sus vecinos izquierdos/inferiores.
    salida[1:-1, 1:-1] = (
        (C[1:-1, 2:] - 2.0 * C[1:-1, 1:-1] + C[1:-1, :-2]) / dx ** 2
        + (C[2:, 1:-1] - 2.0 * C[1:-1, 1:-1] + C[:-2, 1:-1]) / dy ** 2
    )
    return salida


def calcular_adveccion_upwind(C, dx, dy, uv, vv):
    """Calcula las derivadas regresivas de advección para velocidades positivas."""
    # Se reserva un campo nulo para mantener derivada cero en las fronteras.
    salida = np.zeros_like(C, dtype=float)
    # Como ambas velocidades son positivas, se usan los vecinos hacia atrás.
    salida[1:-1, 1:-1] = (
        uv * (C[1:-1, 1:-1] - C[1:-1, :-2]) / dx
        + vv * (C[1:-1, 1:-1] - C[:-2, 1:-1]) / dy
    )
    return salida


def calcular_baricentro(C, X_malla, Y_malla, pesos_simpson):
    """Integra masa y momentos con la regla de Simpson bidimensional."""
    # La masa es la integral aproximada de C sobre toda la superficie.
    masa = float(np.sum(pesos_simpson * C))
    # Si no hay masa, se evita dividir por cero y se devuelve un centro nulo.
    if masa <= np.finfo(float).eps:
        return 0.0, 0.0, 0.0
    # Los momentos divididos por la masa producen las coordenadas del baricentro.
    xc = float(np.sum(pesos_simpson * X_malla * C) / masa)
    yc = float(np.sum(pesos_simpson * Y_malla * C) / masa)
    return masa, xc, yc
