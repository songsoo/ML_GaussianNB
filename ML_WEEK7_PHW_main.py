import pandas as pd
import ML_WEEK7_PHW_auto
data = pd.read_csv("bank-additional-full.csv",sep=';')
ML_WEEK7_PHW_auto.get_result(data)