"""Comparacion grafica de la trayectoria original y sus reconstrucciones."""

from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from .modelado_trayectoria import (
    GRADO_POLINOMIO,
    ROOT,
    RUTA_CSV,
    SALTO_NODOS,
    cargar_datos_y_nodos,
    interpolar_global,
    spline_cubico_natural,
)


RUTA_MAPA = ROOT / "data" / "mapa_satelital_completo.jpg"
DIRECTORIO_SALIDA = ROOT / "resultados" / "resultados_ej2"


def graficar_comparaciones(
    t,
    x,
    y,
    z,
    t_nodos,
    x_nodos,
    y_nodos,
    z_nodos,
    x_global,
    y_global,
    z_global,
    x_spline,
    y_spline,
    z_spline,
    ruta_mapa=RUTA_MAPA,
    directorio_salida=DIRECTORIO_SALIDA,
):
    """Genera y guarda las tres comparaciones solicitadas."""
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    mapa_bgr = cv2.imread(str(ruta_mapa), cv2.IMREAD_COLOR)
    if mapa_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {ruta_mapa}")
    mapa_rgb = cv2.cvtColor(mapa_bgr, cv2.COLOR_BGR2RGB)

    # -- GRAFICO 1: MAPA 2D --
    figura, eje = plt.subplots(figsize=(10, 7))
    eje.imshow(mapa_rgb)
    eje.plot(x, y, color="white", linewidth=1.2, label="Trayectoria con ruido")
    eje.plot(
        x_global,
        y_global,
        color="#e8590c",
        linewidth=2,
        label=f"Global (polinomio grado {GRADO_POLINOMIO})",
    )
    eje.plot(x_spline, y_spline, color="#1c7ed6", linewidth=2, label="Local (Spline)")
    eje.scatter(x_nodos, y_nodos, color="#2b8a3e", s=18, label="Nodos", zorder=3)
    eje.set_title("Trayectoria sobre el mapa")
    eje.set_xlabel("X del mapa (pixeles)")
    eje.set_ylabel("Y del mapa (pixeles)")
    eje.legend()
    figura.tight_layout()
    figura.savefig(directorio_salida / "comparacion_mapa_2d.png", dpi=150)
    plt.close(figura)

    # -- GRAFICO 2: PLANO CARTESIANO --
    x_inicial = x[0]
    y_inicial = y[0]
    x_local = x - x_inicial
    y_local = -(y - y_inicial)
    x_global_local = x_global - x_inicial
    y_global_local = -(y_global - y_inicial)
    x_spline_local = x_spline - x_inicial
    y_spline_local = -(y_spline - y_inicial)
    x_nodos_local = x_nodos - x_inicial
    y_nodos_local = -(y_nodos - y_inicial)

    figura, eje = plt.subplots(figsize=(9, 7))
    eje.plot(
        x_local,
        y_local,
        color="black",
        linewidth=1.2,
        label="Trayectoria con ruido",
    )
    eje.plot(
        x_global_local,
        y_global_local,
        color="#e8590c",
        linewidth=2,
        label=f"Global (polinomio grado {GRADO_POLINOMIO})",
    )
    eje.plot(
        x_spline_local,
        y_spline_local,
        color="#1c7ed6",
        linewidth=2,
        label="Local (Spline)",
    )
    eje.scatter(
        x_nodos_local,
        y_nodos_local,
        color="#2b8a3e",
        s=18,
        label="Nodos",
        zorder=3,
    )
    eje.set_aspect("equal", adjustable="datalim")
    eje.set_title("Trayectoria en el plano cartesiano local")
    eje.set_xlabel("Desplazamiento X (pixeles)")
    eje.set_ylabel("Desplazamiento Y (pixeles)")
    eje.legend()
    eje.grid(alpha=0.25)
    figura.tight_layout()
    figura.savefig(directorio_salida / "comparacion_plano_cartesiano.png", dpi=150)
    plt.close(figura)

    # -- GRAFICO 3: ALTITUD --
    figura, eje = plt.subplots(figsize=(10, 5.5))
    eje.plot(t, z, color="black", linewidth=1.2, label="Trayectoria con ruido")
    eje.plot(
        t,
        z_global,
        color="#e8590c",
        linewidth=2,
        label=f"Global (polinomio grado {GRADO_POLINOMIO})",
    )
    eje.plot(t, z_spline, color="#1c7ed6", linewidth=2, label="Local (Spline)")
    eje.set_title("Perfil de altitud")
    eje.set_xlabel("Tiempo (s)")
    eje.set_ylabel("Altitud relativa Z(t)")
    eje.legend()
    eje.grid(alpha=0.25)
    figura.tight_layout()
    figura.savefig(directorio_salida / "comparacion_altitud.png", dpi=150)
    plt.close(figura)

    return directorio_salida


def main():
    """Calcula el modelado y genera las figuras comparativas."""
    datos = cargar_datos_y_nodos(RUTA_CSV, SALTO_NODOS)
    t, x, y, z, t_nodos, x_nodos, y_nodos, z_nodos = datos
    x_global = interpolar_global(t, t_nodos, x_nodos, GRADO_POLINOMIO)
    y_global = interpolar_global(t, t_nodos, y_nodos, GRADO_POLINOMIO)
    z_global = interpolar_global(t, t_nodos, z_nodos, GRADO_POLINOMIO)
    x_spline = spline_cubico_natural(t, t_nodos, x_nodos)
    y_spline = spline_cubico_natural(t, t_nodos, y_nodos)
    z_spline = spline_cubico_natural(t, t_nodos, z_nodos)

    salida = graficar_comparaciones(
        t,
        x,
        y,
        z,
        t_nodos,
        x_nodos,
        y_nodos,
        z_nodos,
        x_global,
        y_global,
        z_global,
        x_spline,
        y_spline,
        z_spline,
    )
    print(f"Figuras guardadas en: {salida.resolve()}")


if __name__ == "__main__":
    main()
