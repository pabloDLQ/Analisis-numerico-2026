import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import metodo


def adaptar_funcion(funcion):
    def f(x, tipo="comun"):
        return funcion(x)

    return f


def f1(x):
    return ((x - 1.5) ** 4) * math.log(x ** 2 + 1)


def f2(x):
    return (x - 0.75) ** 7


def f3(x):
    return (x ** 11) - 1e-4


def f4(x):
    return math.sin(1.0 / (x + 0.1)) - 0.1


def f5(x):
    return math.tan(x) - x - 1


def f6(x):
    return math.sinh(x * 5) - 100


def f7(x):
    return math.exp(-10 * x) - 1e-5


def f8(x):
    return math.log(x - 2.0 + 1e-15) + 5


def f9(x):
    return (x ** 2 - 1.2) ** 2 - 0.1


def f10(x):
    return math.prod([x - k for k in range(1, 6)]) - 0.01


def f11(x):
    return math.exp(x) - math.cos(x) - 10


def f12(x):
    return math.copysign(abs(x - 2.0) ** (1 / 3), x - 2.0)


def f13(x):
    return (x / (x ** 2 + 1)) - 0.45


def f14(x):
    return x ** 5 - 5 * x ** 3 + 4 * x - 0.5


def f15(x):
    return math.exp(-x) * math.sin(2 * math.pi * x) - 1e-3


def f16(x):
    return math.atan(x - 1250.45) - 0.1


def f17(x):
    return x / (1.0 + abs(x)) - 0.9999


CASOS = [
    ("F1", f1, 1.0, 2.0, 1e-12),
    ("F2", f2, 0.0, 1.0, 1e-12),
    ("F3", f3, 0.0, 1.0, 1e-12),
    ("F4", f4, 0.1, 0.4, 1e-12),
    ("F5", f5, 1.0, 1.5, 1e-12),
    ("F6", f6, 0.5, 2.0, 1e-12),
    ("F7", f7, 0.0, 2.0, 1e-12),
    ("F8", f8, 2.0, 3.0, 1e-12),
    ("F9", f9, 1.0, 1.5, 1e-12),
    ("F10", f10, 4.5, 5.2, 1e-12),
    ("F11", f11, 2.0, 3.5, 1e-12),
    ("F12", f12, 1.0, 4.0, 1e-12),
    ("F13", f13, 0.4, 0.8, 1e-12),
    ("F14", f14, 1.8, 2.3, 1e-12),
    ("F15", f15, 0.2, 0.6, 1e-12),
    ("F16", f16, -10000.0, 10000.0, 1e-12),
    ("F17", f17, -200000.0, 200000.0, 1e-12),
]


def main():
    for nombre, funcion, a, b, tol in CASOS:
        f = adaptar_funcion(funcion)
        try:
            raiz = metodo.encontrar_raiz(f, a, b, tol)
            valor = funcion(raiz)
            print(f"{nombre}: raiz = {raiz:.12f} | f(raiz) = {valor:.3e}")
        except Exception as error:
            print(f"{nombre}: ERROR -> {error}")


if __name__ == "__main__":
    main()