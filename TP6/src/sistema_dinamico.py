"""Definición analítica del sistema dinámico acoplado contaminante-dron.

El modelo se expresa como la ecuación vectorial de primer orden

    dX/dt = F(t, X),

donde X = [xc, yc, sigma, M, xd, yd].
"""

import numpy as np


# Constante matemática pi, utilizada en la expresión temporal de la ráfaga.
PI = np.pi

# Coeficiente de difusión del contaminante, que aparece en d(sigma)/dt = D/sigma.
D = 0.5

# Tasa de decaimiento de la concentración total, que aparece en dM/dt = -k*M.
K = 0.008

# Distancia umbral que separa los regímenes de persecución y captura.
DISTANCIA_CAPTURA = 1.2

# Amplitud de la oscilación transversal de la dirección del dron.
A = 0.75

# Frecuencia angular de la oscilación transversal, en radianes por segundo.
OMEGA = 0.9

# Velocidad del dron relativa a su dirección de movimiento, en unidades por segundo.
VELOCIDAD_DRON = 4.0

# Fracción de la velocidad del viento que arrastra pasivamente al dron.
ALFA = 0.5

# Duración del ciclo temporal utilizado para calcular el viento periódico.
PERIODO_VIENTO = 12.0

# Instante inicial del dominio temporal de integración, en segundos.
T_INICIAL = 0.0

# Instante final del dominio temporal de integración, en segundos.
T_FINAL = 60.0

# Vector inicial X(0) = [xc(0), yc(0), sigma(0), M(0), xd(0), yd(0)].
X0 = np.array([20.0, 20.0, 7.0, 7.0, 100.0, 100.0], dtype=float)


def derivadas_sistema(t, X):
    """Calcula F(t, X), es decir, el vector de derivadas del sistema.

    Parámetros
    ----------
    t : float
        Tiempo actual de la integración, medido en segundos.
    X : array_like
        Estado [xc, yc, sigma, M, xd, yd] en el tiempo t.

    Retorna
    -------
    numpy.ndarray
        Vector [dxc_dt, dyc_dt, dsigma_dt, dM_dt, dxd_dt, dyd_dt].
    """

    # Se convierte la entrada a un array NumPy de tipo real para operar vectorialmente.
    estado = np.asarray(X, dtype=float)

    # El modelo tiene exactamente seis variables de estado.
    if estado.shape != (6,):
        raise ValueError("X debe ser un vector de NumPy de longitud 6.")

    # Se extraen las coordenadas del centro de la nube, su radio y su masa remanente.
    xc, yc, sigma, masa, xd, yd = estado

    # Se reduce el tiempo al ciclo de viento mediante el resto de la división por 12.
    tau = np.mod(t, PERIODO_VIENTO)

    # Se evalúa la ráfaga cuando el tiempo reducido pertenece al intervalo [0, 5).
    if tau < 5.0:
        # Se calcula el factor sinusoidal cuadrático que modula la intensidad de la ráfaga.
        factor_ráfaga = np.sin(PI * tau / 5.0) ** 2

        # Se calcula la componente horizontal del viento durante la ráfaga.
        uv = 0.4 + 2.8 * factor_ráfaga

        # Se calcula la componente vertical del viento durante la ráfaga.
        vv = 0.2 + 2.4 * factor_ráfaga * np.sin(0.6 * t)
    else:
        # En calma, la componente horizontal del viento es constante.
        uv = 0.4

        # En calma, la componente vertical del viento es constante.
        vv = 0.2

    # La nube contaminante se traslada con la velocidad instantánea del viento horizontal.
    dxc_dt = uv

    # La nube contaminante se traslada con la velocidad instantánea del viento vertical.
    dyc_dt = vv

    # El radio de dispersión aumenta según la ley difusiva d(sigma)/dt = D/sigma.
    dsigma_dt = D / sigma

    # La masa contaminante disminuye según la ley de primer orden dM/dt = -K*M.
    dmasa_dt = -K * masa

    # Se construye el vector desde la posición del dron hasta el centro contaminante.
    vector_relativo = np.array([xc - xd, yc - yd], dtype=float)

    # Se calcula la distancia euclídea entre el dron y el centro contaminante.
    distancia_relativa = np.linalg.norm(vector_relativo)

    # Si el dron está capturado, queda arrastrado por el viento.
    if distancia_relativa < DISTANCIA_CAPTURA:
        # La velocidad horizontal del dron coincide con la del viento.
        dxd_dt = uv

        # La velocidad vertical del dron coincide con la del viento.
        dyd_dt = vv
    else:
        # Se normaliza el vector relativo para obtener el versor hacia el objetivo.
        u_hat = vector_relativo / distancia_relativa

        # Se extraen las componentes del versor de persecución.
        ux, uy = u_hat

        # Se construye el versor ortogonal usado para el barrido transversal.
        u_perp = np.array([-uy, ux], dtype=float)

        # Se calcula el vector de orientación oscilante h(t).
        orientacion = u_hat + A * np.sin(OMEGA * t) * u_perp

        # Se calcula la norma de la orientación para convertirla en un versor.
        norma_orientacion = np.linalg.norm(orientacion)

        # Se normaliza la orientación y se multiplica por la velocidad del dron.
        velocidad_dirigida = VELOCIDAD_DRON * orientacion / norma_orientacion

        # Se agrega el arrastre pasivo horizontal producido por el viento.
        dxd_dt = velocidad_dirigida[0] + ALFA * uv

        # Se agrega el arrastre pasivo vertical producido por el viento.
        dyd_dt = velocidad_dirigida[1] + ALFA * vv

    # Se ensamblan todas las derivadas en el mismo orden que las variables de X.
    derivadas = np.array(
        [dxc_dt, dyc_dt, dsigma_dt, dmasa_dt, dxd_dt, dyd_dt],
        dtype=float,
    )

    # Se verifica que la función devuelva explícitamente un vector NumPy de seis entradas.
    if not isinstance(derivadas, np.ndarray) or derivadas.shape != (6,):
        raise RuntimeError("F(t, X) debe ser un array de NumPy de longitud 6.")

    # Se devuelve F(t, X), el lado derecho del sistema de EDOs.
    return derivadas
