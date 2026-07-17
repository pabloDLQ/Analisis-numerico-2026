import unittest

import metodo


def f2(x, tipo="comun"):
    return (x - 0.75) ** 7


class TestMetodo(unittest.TestCase):
    def test_biseccion_fallback_converge_a_raiz(self):
        raiz = metodo.encontrar_raiz(f2, 0.0, 1.0, 1e-12)
        self.assertAlmostEqual(raiz, 0.75, delta=1e-6)

    def test_acepta_argumento_max_iter(self):
        raiz = metodo.encontrar_raiz(f2, 0.0, 1.0, 1e-12, max_iter=50)
        self.assertAlmostEqual(raiz, 0.75, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
