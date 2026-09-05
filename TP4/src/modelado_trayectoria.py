"""Carga de datos y seleccion de nodos para la trayectoria del dron."""

from pathlib import Path

import numpy as np


# =========================================
# PARAMETROS CONFIGURABLES
# =========================================
ROOT = Path(__file__).resolve().parents[1]
RUTA_CSV = ROOT / "resultados" / "resultados_ej1" / "trayectoria_dron.csv"
SALTO_NODOS = 30
GRADO_POLINOMIO = 16


# =========================================
# PASO 1: CARGA DE DATOS Y SELECCION DE NODOS
# =========================================
def cargar_datos_y_nodos(ruta_csv, salto_nodos):
    """Carga la trayectoria completa y selecciona sus nodos de interpolacion."""
    if not isinstance(salto_nodos, (int, np.integer)) or salto_nodos <= 0:
        raise ValueError("salto_nodos debe ser un entero positivo")

    datos = np.genfromtxt(
        ruta_csv,
        delimiter=",",
        skip_header=1,
        usecols=(1, 2, 3, 5),
        dtype=float,
    )
    datos = np.atleast_2d(datos)
    if datos.size == 0 or datos.shape[1] != 4:
        raise ValueError("El CSV no contiene datos validos de trayectoria")

    t, x, y, z = datos.T
    indices_nodos = np.arange(0, len(t), salto_nodos)
    if indices_nodos[-1] != len(t) - 1:
        indices_nodos = np.append(indices_nodos, len(t) - 1)

    t_nodos = t[indices_nodos]
    x_nodos = x[indices_nodos]
    y_nodos = y[indices_nodos]
    z_nodos = z[indices_nodos]

    return t, x, y, z, t_nodos, x_nodos, y_nodos, z_nodos


# =========================================
# PASO 2: METODO GLOBAL (AJUSTE POLINOMIAL)
# =========================================
def interpolar_global(t, t_nodos, var_nodos, grado):
    """Ajusta un polinomio global y lo evalua sobre todos los tiempos."""
    coeficientes = np.polyfit(t_nodos, var_nodos, grado)
    var_global = np.polyval(coeficientes, t)
    error_ini = var_nodos[0] - var_global[0]
    error_fin = var_nodos[-1] - var_global[-1]
    fraccion = (t - t[0]) / (t[-1] - t[0])
    correccion = error_ini * (1.0 - fraccion) + error_fin * fraccion
    return var_global + correccion


# =========================================
# PASO 3: METODO LOCAL (SPLINE CUBICO NATURAL)
# =========================================
def spline_cubico_natural(t, t_nodos, var_nodos):
    """Evalua un spline cubico natural construido desde cero."""
    t = np.asarray(t, dtype=float)
    t_nodos = np.asarray(t_nodos, dtype=float)
    var_nodos = np.asarray(var_nodos, dtype=float)

    if len(t_nodos) < 2:
        raise ValueError("Se necesitan al menos dos nodos")
    if len(t_nodos) != len(var_nodos):
        raise ValueError("Los nodos temporales y sus valores deben tener igual longitud")
    if not np.all(np.isfinite(t_nodos)) or not np.all(np.isfinite(var_nodos)):
        raise ValueError("Los nodos deben contener valores finitos")

    # Calcula la longitud de cada tramo y exige tiempos estrictamente crecientes.
    h = np.diff(t_nodos)
    if np.any(h <= 0):
        raise ValueError("Los tiempos de los nodos deben ser estrictamente crecientes")

    cantidad_nodos = len(t_nodos)
    sistema = np.zeros((cantidad_nodos, cantidad_nodos), dtype=float)
    termino_independiente = np.zeros(cantidad_nodos, dtype=float)

    # Impone las condiciones naturales: derivada segunda nula en ambos extremos.
    sistema[0, 0] = 1.0
    sistema[-1, -1] = 1.0

    # Arma la matriz tridiagonal y el segundo miembro en los nodos interiores.
    pendientes = np.diff(var_nodos) / h
    for indice in range(1, cantidad_nodos - 1):
        sistema[indice, indice - 1] = h[indice - 1]
        sistema[indice, indice] = 2.0 * (h[indice - 1] + h[indice])
        sistema[indice, indice + 1] = h[indice]
        termino_independiente[indice] = 6.0 * (
            pendientes[indice] - pendientes[indice - 1]
        )

    # Resuelve las derivadas segundas nodales del spline.
    derivadas_segundas = np.linalg.solve(sistema, termino_independiente)
    valores = np.empty_like(t, dtype=float)

    # Evalua la formula cubica correspondiente en cada tramo temporal.
    valores[:] = np.nan
    for indice in range(cantidad_nodos - 1):
        if indice == cantidad_nodos - 2:
            mascara = (t >= t_nodos[indice]) & (t <= t_nodos[indice + 1])
        else:
            mascara = (t >= t_nodos[indice]) & (t < t_nodos[indice + 1])
        distancia = t[mascara] - t_nodos[indice]
        a = (t_nodos[indice + 1] - t[mascara]) / h[indice]
        b = distancia / h[indice]
        valores[mascara] = (
            a * var_nodos[indice]
            + b * var_nodos[indice + 1]
            + (
                (a**3 - a) * derivadas_segundas[indice]
                + (b**3 - b) * derivadas_segundas[indice + 1]
            )
            * h[indice] ** 2
            / 6.0
        )

    if np.any(~np.isfinite(valores)):
        raise ValueError("El vector t debe estar dentro del intervalo de los nodos")
    return valores


# =========================================
# EJECUCION DEL MODELADO
# =========================================
if __name__ == "__main__":
    # Configuracion del experimento: maxima suavidad, un nodo cada 2 segundos.
    SALTO_NODOS = 30
    GRADO_GLOBAL = 5
    RUTA_CSV = ROOT / "resultados" / "resultados_ej1" / "trayectoria_dron.csv"

    resultado = cargar_datos_y_nodos(RUTA_CSV, SALTO_NODOS)
    t, _, _, _, t_nodos, x_nodos, y_nodos, z_nodos = resultado
    x_global = interpolar_global(t, t_nodos, x_nodos, GRADO_GLOBAL)
    y_global = interpolar_global(t, t_nodos, y_nodos, GRADO_GLOBAL)
    z_global = interpolar_global(t, t_nodos, z_nodos, GRADO_GLOBAL)
    x_spline = spline_cubico_natural(t, t_nodos, x_nodos)
    y_spline = spline_cubico_natural(t, t_nodos, y_nodos)
    z_spline = spline_cubico_natural(t, t_nodos, z_nodos)

    print(f"Frames completos: {len(t)}")
    print(f"Nodos seleccionados: {len(t_nodos)}")
    print(f"Grado del polinomio global: {GRADO_GLOBAL}")
    print(f"Curvas globales calculadas: {len(x_global)}, {len(y_global)}, {len(z_global)}")
    print(f"Curvas spline calculadas: {len(x_spline)}, {len(y_spline)}, {len(z_spline)}")
    print(f"Primer frame: t = {t_nodos[0]}")
    print(f"Ultimo frame: t = {t_nodos[-1]}")
