import pandas as pd


def addition(a, b):
    return a + b


def test_addition():
    assert addition(2, 3) == 5


def test_pas_de_valeurs_manquantes():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert df.isnull().sum().sum() == 0
