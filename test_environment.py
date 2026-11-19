"""Vérifie que l'environnement Python est correctement installé."""
import numpy as np
import pandas as pd
import sklearn
import mlflow

print("Environment is set up correctly!")
print("numpy", np.__version__, "| pandas", pd.__version__,
      "| sklearn", sklearn.__version__, "| mlflow", mlflow.__version__)
