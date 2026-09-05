"""Registro geométrico del video contra el mapa satelital.

Este modulo usa ORB + matching por fuerza bruta + homografia robusta con
RANSAC para estimar la pose planar del frame respecto del mapa. La razon es
que el movimiento de la camara se ha de interpretar en coordenadas del mapa, no
solo en pixels del video, para reconstruir la trayectoria global.

La funcion principal intenta alinear cada frame con el mapa usando puntos clave
repetibles y descarta coincidencias ambiguas para no introducir errores de
proyeccion en la trayectoria final.
"""

from __future__ import annotations

import cv2
import numpy as np


def register_frame(frame_gray, map_gray, orb, map_keypoints, map_descriptors):
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
    if descriptors is None or len(keypoints) < 8:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, map_descriptors, k=2)
    good = [a for a, b in matches if a.distance < 0.72 * b.distance]
    if len(good) < 8:
        return None, 0
    source = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([map_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5)
    inliers = int(mask.sum()) if mask is not None else 0
    return (homography, inliers) if homography is not None and inliers >= 8 else (None, 0)
