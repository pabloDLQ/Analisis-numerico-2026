import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import metodo


class TelemetriaLlamadas:
    def __init__(self, funcion):
        self.funcion = funcion
        self.llamadas = []

    def __call__(self, x, tipo="comun"):
        self.llamadas.append(tipo)
        return self.funcion(x)

    def puntaje(self):
        puntaje = 0
        tipo_anterior = None
        racha = 0

        for tipo in self.llamadas:
            if tipo == tipo_anterior:
                racha += 1
            else:
                tipo_anterior = tipo
                racha = 1

            if tipo == "comun":
                puntaje += 4 if racha > 4 else 1
            elif tipo == "derivada":
                puntaje += 4 if racha > 4 else 2
            else:
                raise ValueError(f"Tipo de llamada desconocido: {tipo}")

        return puntaje


def adaptar_funcion(funcion):
    return TelemetriaLlamadas(funcion)


def calcular_penalizacion_resultado(raiz, raiz_esperada, error_maximo=1e-7):
    error_absoluto = abs(raiz - raiz_esperada)
    if error_absoluto <= error_maximo:
        return 0, error_absoluto

    return 1000, error_absoluto


def es_colapso_por_iteraciones(telemetria, penalizacion_resultado, error_absoluto):
    return penalizacion_resultado == 1000 and error_absoluto > 1e-7 and len(telemetria.llamadas) >= 300


def calcular_raiz_referencia(nombre, funcion, a, b, semilla=None):
    if nombre == "F1":
        return 1.5
    if nombre == "F2":
        return 0.75
    if nombre == "F3":
        return 1e-4 ** (1 / 11)
    if nombre == "F6":
        return math.asinh(100.0) / 5.0
    if nombre == "F7":
        return math.log(1e5) / 10.0
    if nombre == "F8":
        return 2.0 - 1e-15 + math.exp(-5.0)
    if nombre == "F12":
        return 2.0
    if nombre == "F16":
        return 1250.45 + math.tan(0.1)
    if nombre == "F17":
        return 9999.0

    tolerancia = 1e-15
    fa = funcion(a)
    fb = funcion(b)

    if abs(fa) <= tolerancia:
        return a
    if abs(fb) <= tolerancia:
        return b

    if fa * fb > 0:
        if semilla is not None:
            return semilla
        return (a + b) / 2.0

    low = a
    high = b
    f_low = fa
    f_high = fb

    for _ in range(250):
        medio = (low + high) / 2.0
        f_medio = funcion(medio)

        if abs(f_medio) <= tolerancia or (high - low) <= tolerancia:
            return medio

        if f_low * f_medio <= 0:
            high = medio
            f_high = f_medio
        else:
            low = medio
            f_low = f_medio

    return (low + high) / 2.0


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
    ("F1", f1, 1.0, 2.0, 1e-12,1.5000000),
    ("F2", f2, 0.0, 1.0, 1e-12,0.7500000),
    ("F3", f3, 0.0, 1.0, 1e-12,0.4328761),
    ("F4", f4, 0.1, 0.4, 1e-12,0.2287932),
    ("F5", f5, 1.0, 1.5, 1e-12,1.1322677),
    ("F6", f6, 0.5, 2.0, 1e-12,1.3958154),
    ("F7", f7, 0.0, 2.0, 1e-12,1.1512926),
    ("F8", f8, 2.0, 3.0, 1e-12,2.0067379),
    ("F9", f9, 1.0, 1.5, 1e-12,1.2313520),
    ("F10", f10, 4.5, 5.2, 1e-12,5.0004163),
    ("F11", f11, 2.0, 3.5, 1e-12,2.2386538),
    ("F12", f12, 1.0, 4.0, 1e-12,2.0000000),
    ("F13", f13, 0.4, 0.8, 1e-12,0.6267890),
    ("F14", f14, 1.8, 2.3, 1e-12,2.0199891),
    ("F15", f15, 0.2, 0.6, 1e-12,0.4997375),
    ("F16", f16, -10000.0, 10000.0, 1e-12,1250.5503349),
    ("F17", f17, -200000.0, 200000.0, 1e-12,9999.0000000),
]


def main():
    puntaje_total = 0

    for nombre, funcion, a, b, tol, raiz_semilla in CASOS:
        raiz_esperada = calcular_raiz_referencia(nombre, funcion, a, b, raiz_semilla)
        f = adaptar_funcion(funcion)
        try:
            raiz = metodo.encontrar_raiz(f, a, b, tol)
            valor = funcion(raiz)
            puntaje_llamadas = f.puntaje()
            penalizacion_resultado, error_absoluto = calcular_penalizacion_resultado(
                raiz, raiz_esperada
            )
            if es_colapso_por_iteraciones(f, penalizacion_resultado, error_absoluto):
                penalizacion_resultado = 1500
            puntaje_caso = puntaje_llamadas + penalizacion_resultado
            puntaje_total += puntaje_caso

            print(
                f"{nombre}: raiz = {raiz:.12f} | f(raiz) = {valor:.3e} | "
                f"puntaje = {puntaje_caso}"
            )

            if penalizacion_resultado > 0:
                if penalizacion_resultado == 1500:
                    print(f"{nombre}: +{penalizacion_resultado} por colapso del sistema")
                else:
                    print(
                        f"{nombre}: +{penalizacion_resultado} por raiz incorrecta "
                        f"(error = {error_absoluto:.3e})"
                    )
        except Exception as error:
            puntaje_llamadas = f.puntaje()
            penalizacion_resultado = 1500
            puntaje_caso = puntaje_llamadas + penalizacion_resultado
            puntaje_total += puntaje_caso

            print(f"{nombre}: ERROR -> {error}")
            print(f"{nombre}: +{penalizacion_resultado} por colapso del sistema")

    print(f"PUNTAJE FINAL = {puntaje_total}")


if __name__ == "__main__":
    main()