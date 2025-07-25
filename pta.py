import pandas as pd
import pandas_ta as ta

# Crea un DataFrame vacío para poder usar la extensión
df = pd.DataFrame()

# Imprime la lista de todos los indicadores disponibles
print(df.ta.indicators())
