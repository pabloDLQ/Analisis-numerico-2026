import unittest

import metodo


def f2(x, tipo="comun"):
    return (x - 0.75) ** 7


class TestMetodo(unittest.TestCase):
    def test_encontrar_raiz_converge_a_raiz(self):
        raiz = metodo.encontrar_raiz(f2, 0.0, 1.0, 1e-12)
        self.assertAlmostEqual(raiz, 0.75, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
