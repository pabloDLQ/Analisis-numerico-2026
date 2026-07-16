def encontrar_raiz(f, a, b, tol, max_iter=150):
    """
    Argumentos de entrada:
    f : Función a evaluar. Se debe invocar especificando el "tipo" de llamada:
    - Para evaluar la función de forma estándar: f(x, tipo="comun")
    - Para evaluar puntos destinados al cálculo de derivadas: f(x, tipo="derivada")
    a, b: Flotantes. Extremos del intervalo inicial (b > a).
    tol : Flotante>0. Tolerancia de convergencia.
    El algoritmo DEBE detenerse estrictamente cuando | x_k - x_{k-1} | <= tol.
    Para la evaluación, el equipo docente utilizará: tol = 1e-12
    max_iter : Entero opcional. Cantidad máxima de iteraciones permitidas.
    Retorno:
    Devuelve únicamente un número flotante con la estimación de la raíz.
    """
    # Ejemplo de llamadas válidas para la telemetría del peaje (sistema de benchmark):
    # fx = f(x, tipo="comun")
    # fx_der = f(x + dx, tipo="derivada")

    fa = f(a, tipo="comun")
    fb = f(b, tipo="comun")

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    if fa * fb > 0:
        raise ValueError("El intervalo no contiene un cambio de signo")

    # Mantenemos |fb| <= |fa| para que b sea el mejor extremo actual.
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = c
    mflag = True
    x_anterior = b

    for _ in range(max_iter):
        if fa != fc and fb != fc:
            denom0 = (fa - fb) * (fa - fc)
            denom1 = (fb - fa) * (fb - fc)
            denom2 = (fc - fa) * (fc - fb)

            if denom0 != 0.0 and denom1 != 0.0 and denom2 != 0.0:
                s = (
                    a * (fb * fc) / denom0
                    + b * (fa * fc) / denom1
                    + c * (fa * fb) / denom2
                )
            else:
                s = b - fb * (b - a) / (fb - fa)
        else:
            s = b - fb * (b - a) / (fb - fa)

        limite = (3.0 * a + b) / 4.0
        lower = min(limite, b)
        upper = max(limite, b)
        cond1 = not (lower < s < upper)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = f(s, tipo="comun")

        if fs == 0.0:
            return s

        if abs(s - x_anterior) <= tol:
            return s

        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(b - a) <= tol:
            return b

        x_anterior = s

    raise RuntimeError("No se alcanzó la tolerancia en el máximo de iteraciones")