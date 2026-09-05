"""Reconstruccion de la escala relativa Z(t).

La escala se acumula multiplicativamente como producto de los factores de
escala locales entre frames. Para evitar que pequenos errores del flujo optico
se conviertan en drift grande, este modulo trabaja en log-escala,
descarta outliers con MAD y aplica suavizado temporal conservador.

La idea es transformar los cambios relativos de escala en una curva continua y
estable que represente la variacion de altitud o zoom aparente del dron a lo
largo del video, sin dejar que el ruido puntal domine la sensacion global.
"""

from __future__ import annotations

import numpy as np

from .datos_trayectoria import Trajectory


def _rebuild_relative_scale(trajectory: Trajectory):
    steps = np.clip(trajectory.scale_step.astype(float).copy(), 0.992, 1.008)
    errors = trajectory.lk_error.astype(float)
    valid_points = trajectory.valid_points.astype(int)

    log_steps = np.log(steps)
    core = log_steps[1:] if len(log_steps) > 1 else log_steps
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)))
    spread = max(1e-5, 1.4826 * mad)
    threshold = 2.5 * spread
    error_cut = float(np.percentile(errors[np.isfinite(errors)], 70))

    noisy = ((np.abs(log_steps - median) > threshold) & (errors >= error_cut)) | (valid_points < 20)
    log_steps[noisy] = median

    log_steps = np.clip(log_steps, median - threshold, median + threshold)

    window = 21
    kernel = np.ones(window, dtype=float) / window
    log_smoothed = np.convolve(log_steps, kernel, mode="same")
    log_smoothed[0] = 0.0

    z = np.exp(np.cumsum(log_smoothed))
    z /= z[0]

    # Se elimina la deriva lenta para conservar la variacion local sin arrastrar
    # una tendencia de fondo que no corresponde al movimiento real del dron.
    log_z = np.log(z)
    drift = np.linspace(log_z[0], log_z[-1], len(log_z))
    z = np.exp(log_z - drift + log_z[0])
    z /= z[0]
    return z
