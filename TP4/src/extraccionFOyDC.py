"""Trayectoria del dron sobre el mapa usando Lucas-Kanade explicito."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data" / "vuelo_dron.mp4"
DEFAULT_MAP = ROOT / "data" / "mapa_satelital_completo.jpg"
DEFAULT_OUTPUT = ROOT / "data" / "resultados"


@dataclass
class Trajectory:
    frame: np.ndarray
    time_s: np.ndarray
    x_map: np.ndarray
    y_map: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    scale_step: np.ndarray
    valid_points: np.ndarray
    lk_error: np.ndarray
    map_inliers: np.ndarray
    registered: np.ndarray


def detect_features(gray: np.ndarray, max_corners: int = 300) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 8, blockSize=7)
    return np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)


def lucas_kanade(previous: np.ndarray, current: np.ndarray, points: np.ndarray, radius: int = 5):
    """Calcula cada vector resolviendo (A.T A) u = A.T b.

    A contiene Ix e Iy y b=G-F, es decir, la ecuacion de Taylor
    Ix*u + Iy*v = G-F indicada en el enunciado.
    """
    ix = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=3)
    temporal = current.astype(np.float32) - previous.astype(np.float32)
    height, width = previous.shape
    old_valid, new_valid, flows, errors = [], [], [], []
    for point in points.reshape(-1, 2):
        x, y = np.round(point).astype(int)
        x0, x1, y0, y1 = x - radius, x + radius + 1, y - radius, y + radius + 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        matrix = np.column_stack((ix[y0:y1, x0:x1].ravel(), iy[y0:y1, x0:x1].ravel()))
        difference = temporal[y0:y1, x0:x1].ravel()
        normal = matrix.T @ matrix
        if np.linalg.det(normal) <= 1e-3:
            continue
        flow = np.linalg.solve(normal, matrix.T @ difference)
        if not np.isfinite(flow).all() or np.linalg.norm(flow) > 20:
            continue
        old_valid.append(point)
        new_valid.append(point + flow)
        flows.append(flow)
        errors.append(np.sqrt(np.mean((matrix @ flow - difference) ** 2)))
    return (np.asarray(old_valid, np.float32).reshape(-1, 2),
            np.asarray(new_valid, np.float32).reshape(-1, 2),
            np.asarray(flows, np.float32).reshape(-1, 2),
            float(np.median(errors)) if errors else float("inf"))


def robust_mask(flows: np.ndarray) -> np.ndarray:
    if len(flows) == 0:
        return np.zeros(0, dtype=bool)
    distances = np.linalg.norm(flows - np.median(flows, axis=0), axis=1)
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    return distances <= max(2.5, median + 3 * 1.4826 * mad)


def relative_scale(old: np.ndarray, new: np.ndarray) -> float:
    if len(old) < 3:
        return 1.0
    old_center, new_center = np.median(old, axis=0), np.median(new, axis=0)
    old_radius = np.linalg.norm(old - old_center, axis=1)
    new_radius = np.linalg.norm(new - new_center, axis=1)
    usable = old_radius > 1
    ratios = new_radius[usable] / old_radius[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.5) & (ratios < 2)]
    return float(np.median(ratios)) if len(ratios) else 1.0


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


def extract_trajectory(video_path=DEFAULT_VIDEO, map_path=DEFAULT_MAP, max_corners=300, registration_step=5):
    map_gray = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if map_gray is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {map_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError("El video no contiene frames legibles")
    previous = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    points = detect_features(previous, max_corners)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    map_keypoints, map_descriptors = orb.detectAndCompute(map_gray, None)
    if map_descriptors is None:
        capture.release()
        raise ValueError("No se encontraron caracteristicas en el mapa")
    image_height, image_width = previous.shape
    center = np.float32([[[image_width / 2, image_height / 2]]])
    map_position = np.array([np.nan, np.nan])
    start_position = None
    homography = None
    rows = [[0, 0.0, np.nan, np.nan, 1.0, 0, 0, 1.0, len(points), 0, 0, 0]]
    z_value = 1.0
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        current = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        old_valid, new_valid, flows, lk_error = lucas_kanade(previous, current, points)
        mask = robust_mask(flows)
        if mask.sum() >= 8:
            old_inliers, new_inliers = old_valid[mask], new_valid[mask]
            flow = np.median(flows[mask], axis=0)
            scale_step = relative_scale(old_inliers, new_inliers)
            z_value *= scale_step
            u, v = map(float, flow)
            points = new_inliers
        else:
            old_inliers, new_inliers = old_valid[:0], new_valid[:0]
            scale_step, u, v = 1.0, 0.0, 0.0
        if len(points) < 25:
            points = detect_features(current, max_corners)
        map_inliers = 0
        if frame_number % registration_step == 0 or homography is None:
            new_homography, map_inliers = register_frame(current, map_gray, orb, map_keypoints, map_descriptors)
            if new_homography is not None:
                homography = new_homography
        if homography is not None:
            projected = cv2.perspectiveTransform(center, homography)[0, 0]
            if np.isfinite(projected).all():
                map_position = projected.astype(float)
                if start_position is None:
                    start_position = map_position.copy()
        rows.append([frame_number, frame_number / fps, map_position[0], map_position[1], z_value, u, -v, scale_step, len(new_inliers), lk_error, map_inliers, int(homography is not None)])
        previous = current
    capture.release()
    data = np.asarray(rows, float)
    if start_position is None:
        raise RuntimeError("No se pudo registrar ningun frame contra el mapa")
    # Impone el requisito fisico de aterrizar en el punto inicial, corrigiendo la deriva.
    data[0, 2:4] = start_position
    closure = data[-1, 2:4] - start_position
    data[:, 2:4] -= np.linspace(0, 1, len(data))[:, None] * closure
    return Trajectory(*[data[:, i].astype(dtype) for i, dtype in enumerate([int, float, float, float, float, float, float, float, int, float, int, int])])


def _fix_xy_outliers(trajectory: Trajectory, jump_threshold: float = 55.0, min_inliers: int = 18):
    x = trajectory.x_map.astype(float).copy()
    y = trajectory.y_map.astype(float).copy()
    inliers = trajectory.map_inliers.astype(int)

    jumps = np.hypot(np.diff(x), np.diff(y))
    bad = np.zeros(len(x), dtype=bool)

    # Descarta saltos bruscos en frames de baja confianza de registro.
    if len(jumps) > 0:
        conflict = np.where((jumps > jump_threshold) & (inliers[1:] < min_inliers))[0] + 1
        bad[conflict] = True

    # Filtra cambios de posicion anormalmente altos para estabilizar tramos ruidosos.
    if len(jumps) > 10:
        base = float(np.median(jumps))
        mad = float(np.median(np.abs(jumps - base)))
        dynamic_limit = max(jump_threshold, base + 4.0 * 1.4826 * mad)
        high = np.where(jumps > dynamic_limit)[0] + 1
        bad[high] = True

    # Limpia picos aislados (salto y retorno inmediato).
    for i in range(1, len(x) - 1):
        left = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        right = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        bridge = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        if left > 35.0 and right > 35.0 and bridge < 25.0:
            bad[i] = True

    # Si un frame tiene baja confianza y esta rodeado de saltos, se descarta.
    for i in range(1, len(x) - 1):
        left = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        right = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        if inliers[i] < min_inliers and (left > 35.0 or right > 35.0):
            bad[i] = True

    bad[0] = False
    bad[-1] = False
    good = ~bad
    if good.sum() < 2:
        return x, y

    idx = np.arange(len(x), dtype=float)
    x = np.interp(idx, idx[good], x[good])
    y = np.interp(idx, idx[good], y[good])

    # Filtro mediano + suavizado para eliminar dientes causados por homografias inestables.
    for _ in range(2):
        x_pad = np.pad(x, (3, 3), mode="edge")
        y_pad = np.pad(y, (3, 3), mode="edge")
        x = np.array([np.median(x_pad[i:i + 7]) for i in range(len(x))], dtype=float)
        y = np.array([np.median(y_pad[i:i + 7]) for i in range(len(y))], dtype=float)

    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    x_pad = np.pad(x, (3, 3), mode="edge")
    y_pad = np.pad(y, (3, 3), mode="edge")
    x = np.convolve(x_pad, kernel, mode="valid")
    y = np.convolve(y_pad, kernel, mode="valid")
    return x, y


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

    # Elimina deriva de baja frecuencia sin alterar las variaciones locales.
    log_z = np.log(z)
    drift = np.linspace(log_z[0], log_z[-1], len(log_z))
    z = np.exp(log_z - drift + log_z[0])
    z /= z[0]
    return z


def postprocess_trajectory(trajectory: Trajectory) -> Trajectory:
    x_fixed, y_fixed = _fix_xy_outliers(trajectory)
    z_fixed = _rebuild_relative_scale(trajectory)
    return Trajectory(
        frame=trajectory.frame,
        time_s=trajectory.time_s,
        x_map=x_fixed,
        y_map=y_fixed,
        z=z_fixed,
        u=trajectory.u,
        v=trajectory.v,
        scale_step=trajectory.scale_step,
        valid_points=trajectory.valid_points,
        lk_error=trajectory.lk_error,
        map_inliers=trajectory.map_inliers,
        registered=trajectory.registered,
    )


def save_results(trajectory, map_path=DEFAULT_MAP, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trayectoria_mapa.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "tiempo_s", "X_mapa_px", "Y_mapa_px", "Z_relativa", "u_LK", "v_LK", "escala_frame", "puntos_validos", "error_LK", "inliers_mapa", "registrado"])
        writer.writerows(zip(trajectory.frame, trajectory.time_s, trajectory.x_map, trajectory.y_map, trajectory.z, trajectory.u, trajectory.v, trajectory.scale_step, trajectory.valid_points, trajectory.lk_error, trajectory.map_inliers, trajectory.registered))
    y_cartesian = -trajectory.y_map
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(trajectory.x_map, y_cartesian, color="#e8590c")
    axis.set_aspect("equal")
    axis.set_title("Trayectoria X-Y sobre el mapa satelital (eje Y cartesiano)")
    axis.set_xlabel("X en mapa (pixeles)"); axis.set_ylabel("Y cartesiana (pixeles)"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output_dir / "trayectoria_xy_mapa.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(trajectory.time_s, trajectory.z, color="#2b8a3e")
    axis.set_title("Escala de altitud/zoom relativo Z(t)")
    axis.set_xlabel("Tiempo (s)"); axis.set_ylabel("Z(t) relativa"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output_dir / "escala_z.png", dpi=150); plt.close(figure)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--mapa", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paso-registro", type=int, default=5)
    args = parser.parse_args()
    trajectory = extract_trajectory(args.video, args.mapa, registration_step=args.paso_registro)
    trajectory = postprocess_trajectory(trajectory)
    csv_path = save_results(trajectory, args.mapa, args.salida)
    print(f"Frames procesados: {len(trajectory.frame)}")
    print(f"Frames con registro de mapa: {int(trajectory.registered.sum())}")
    print(f"Cierre: {np.linalg.norm(trajectory.x_map[-1:] - trajectory.x_map[:1]) + np.linalg.norm(trajectory.y_map[-1:] - trajectory.y_map[:1]):.3f} pixeles")
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
