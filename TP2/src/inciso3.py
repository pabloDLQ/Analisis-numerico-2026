import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from src.cargar_imagenes_jpg import cargar_imagen


def remuestrear_a_log_polar(espectro_magnitud, num_radios=None, num_angulos=360, r_min=1.0):
    """
    Remuestrea un espectro (centrado) a coordenadas log-polares.

    Args:
        espectro_magnitud (np.ndarray): Imagen 2D del espectro (ya fftshift y abs)
        num_radios (int): número de muestras en la dirección radial (default: radio_max)
        num_angulos (int): número de muestras angulares (default: 360)
        r_min (float): radio mínimo para evitar log(0)

    Returns:
        tuple: (espectro_log_polar, radios_exp, delta_log)
            - espectro_log_polar: array shape (num_radios, num_angulos)
            - radios_exp: array de radios usados (exp de la grilla log)
            - delta_log: incremento en log(r) entre filas consecutivas
    """

    h, w = espectro_magnitud.shape
    cy, cx = h / 2.0, w / 2.0
    radio_max = min(cy, cx)

    if num_radios is None:
        num_radios = int(radio_max)

    # Evitar r_min muy pequeño
    r_min = max(1.0, float(r_min))
    if r_min >= radio_max:
        r_min = 1.0

    # Grilla logaritmica en radio
    log_r_min = np.log(r_min)
    log_r_max = np.log(radio_max)
    radios_log = np.linspace(log_r_min, log_r_max, num_radios)
    radios_exp = np.exp(radios_log)

    # Ángulos uniformes 0..2pi
    angulos = np.linspace(0, 2 * np.pi, num_angulos, endpoint=False)

    theta_grid, rho_grid = np.meshgrid(angulos, radios_exp)

    x = cx + rho_grid * np.cos(theta_grid)
    y = cy + rho_grid * np.sin(theta_grid)

    # Remuestrear usando map_coordinates (y, x) en ese orden
    espectro_log_polar = ndimage.map_coordinates(
        espectro_magnitud, [y, x], order=1, mode='constant', cval=0.0
    )

    delta_log = (log_r_max - log_r_min) / max(1, (num_radios - 1))

    return espectro_log_polar, radios_exp, delta_log


def calcular_factor_escala(num_imagen1=1, num_imagen2=5, num_radios=None, num_angulos=360):
    """
    Estima el factor de escala entre dos imágenes usando FFT + remuestreo log-polar
    y correlación de fase.

    Devuelve un diccionario con el factor estimado y métricas auxiliares.
    """

    # Cargar y convertir a gris
    img1 = cargar_imagen(num_imagen1)
    img2 = cargar_imagen(num_imagen2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # FFT, shift y magnitud
    F1 = np.fft.fftshift(np.fft.fft2(gray1))
    F2 = np.fft.fftshift(np.fft.fft2(gray2))

    M1 = np.abs(F1)
    M2 = np.abs(F2)

    # Log para estabilizar rango dinámico
    M1_log = np.log1p(M1)
    M2_log = np.log1p(M2)

    # Remuestrear a log-polar
    LP1, radios1, delta_log = remuestrear_a_log_polar(M1_log, num_radios=num_radios, num_angulos=num_angulos)
    LP2, radios2, _ = remuestrear_a_log_polar(M2_log, num_radios=num_radios, num_angulos=num_angulos)

    # Normalizar cada mapa
    def normalizar(a):
        b = a - np.mean(a)
        s = np.std(b)
        if s <= 1e-10:
            return b
        return b / s

    LP1_n = normalizar(LP1).astype(np.float32)
    LP2_n = normalizar(LP2).astype(np.float32)

    # Aplicar ventana Hann 2D para reducir efectos de borde
    win_y = np.hanning(LP1_n.shape[0])
    win_x = np.hanning(LP1_n.shape[1])
    window = np.outer(win_y, win_x).astype(np.float32)

    LP1_win = LP1_n * window
    LP2_win = LP2_n * window

    # Correlación de fase (OpenCV devuelve shift (dx, dy) y la respuesta)
    shift, response = cv2.phaseCorrelate(LP1_win, LP2_win)
    dx, dy = shift  # dx: columnas (ángulo), dy: filas (radio log)

    # Correlación de fase explícita para conservarla y visualizarla
    espectro_cruzado = np.fft.fft2(LP1_win) * np.conj(np.fft.fft2(LP2_win))
    magnitud_espectro = np.abs(espectro_cruzado)
    magnitud_espectro[magnitud_espectro == 0] = 1
    espectro_cruzado_normalizado = espectro_cruzado / magnitud_espectro
    matriz_correlacion = np.abs(
        np.fft.fftshift(np.fft.ifft2(espectro_cruzado_normalizado))
    )

    # Interpretación: un desplazamiento positivo en filas (dy) indica que LP2
    # está desplazado hacia abajo respecto a LP1 en la dimensión log-radio.
    # Dado que en coordenadas logarítmicas ln(r') = ln(r) + ln(k_effective),
    # el desplazamiento en filas * delta_log = ln(k_spectrum), por lo que
    # k_spectrum = exp(dy * delta_log). Dependiendo de la convención de escala
    # entre dominio espacial y espectral, puede aparecer la inversa.

    # Convertir desplazamiento a factor k en el espectro
    ln_k = dy * delta_log
    k_spectral = float(np.exp(ln_k))

    # Debido a la propiedad de escala: si una imagen espacial se escala por k,
    # su espectro se escala por 1/k. Por eso la estimación espacial es la inversa.
    k_espacial = float(1.0 / k_spectral) if k_spectral != 0 else float('inf')
    f1_escalada = cv2.resize(
        gray1,
        None,
        fx=k_espacial,
        fy=k_espacial,
        interpolation=cv2.INTER_LINEAR
    )

    # Medir confianza (normalizar response a [0,1] aprox.)
    confianza = float(min(1.0, response))

    resultado = {
        'factor_escala': k_espacial,
        'factor_escala_spectral': k_spectral,
        'desplazamiento_log_radio': float(dy),
        'ln_k': float(ln_k),
        'pico_correlacion': float(response),
        'confianza': confianza,
        'metodo': 'FFT -> log-polar -> correlación de fase',
        'log_polar1': LP1_n,
        'log_polar2': LP2_n,
        'matriz_correlacion': matriz_correlacion,
        'f5_original': gray2,
        'f1_escalada': f1_escalada
    }

    return resultado


def guardar_grafico_escala(resultado):
    """
    Guarda un gráfico de cuatro paneles con el análisis del factor de escala.

    Args:
        resultado (dict): Resultado de calcular_factor_escala.
    """
    figura, ejes = plt.subplots(2, 2, figsize=(12, 10))

    ejes[0, 0].imshow(
        resultado['log_polar1'],
        cmap='gray',
        aspect='auto'
    )
    ejes[0, 0].set_title(r'Espectro Log-Polar ($f_1$)')
    ejes[0, 0].set_xlabel(r'Ángulo $\theta$')
    ejes[0, 0].set_ylabel(r'Log-Radio $\rho$')

    ejes[0, 1].imshow(
        resultado['log_polar2'],
        cmap='gray',
        aspect='auto'
    )
    ejes[0, 1].set_title(r'Espectro Log-Polar ($f_5$)')
    ejes[0, 1].set_xlabel(r'Ángulo $\theta$')
    ejes[0, 1].set_ylabel(r'Log-Radio $\rho$')

    matriz_correlacion = resultado['matriz_correlacion']
    indice_theta = np.unravel_index(
        np.argmax(matriz_correlacion),
        matriz_correlacion.shape
    )[1]
    corte_radial = matriz_correlacion[:, indice_theta]
    desplazamientos_radiales = (
        np.arange(len(corte_radial)) - len(corte_radial) // 2
    )
    desplazamiento_log_radio = resultado['desplazamiento_log_radio']
    indice_pico = np.argmin(
        np.abs(desplazamientos_radiales - desplazamiento_log_radio)
    )

    ejes[1, 0].plot(desplazamientos_radiales, corte_radial)
    ejes[1, 0].axvline(
        desplazamiento_log_radio,
        color='red',
        linestyle=':',
        label=f'Pico: {desplazamiento_log_radio:.2f}'
    )
    ejes[1, 0].scatter(
        desplazamientos_radiales[indice_pico],
        corte_radial[indice_pico],
        color='red',
        zorder=3
    )
    ejes[1, 0].set_title('Correlación de Fase (Eje Radial)')
    ejes[1, 0].set_xlabel(r'Desplazamiento $\Delta\rho$')
    ejes[1, 0].set_ylabel('Correlación')
    ejes[1, 0].legend()

    f5_original = resultado['f5_original']
    f1_escalada = resultado['f1_escalada']
    alto = min(f5_original.shape[0], f1_escalada.shape[0])
    ancho = min(f5_original.shape[1], f1_escalada.shape[1])
    f5_recortada = f5_original[:alto, :ancho]
    f1_escalada_recortada = f1_escalada[:alto, :ancho]
    mitad = ancho // 2
    comparacion = np.hstack((
        f5_recortada[:, :mitad],
        f1_escalada_recortada[:, mitad:]
    ))

    ejes[1, 1].imshow(comparacion, cmap='gray')
    ejes[1, 1].set_title(
        r'Validación: $f_5$ Original (Izq) vs $f_1$ Escalada (Der)'
    )
    ejes[1, 1].axis('off')

    figura.tight_layout()
    ruta_salida = Path(__file__).resolve().parent.parent / 'imagenes creadas'
    ruta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = ruta_salida / 'inciso3_analisis_escala.png'
    figura.savefig(archivo_salida, bbox_inches='tight')
    plt.close(figura)

    print(f'Gráfico de análisis de escala guardado en: {archivo_salida}')


if __name__ == '__main__':
    # prueba rápida
    res = calcular_factor_escala(1, 5)
    print('Resultado:', res)
