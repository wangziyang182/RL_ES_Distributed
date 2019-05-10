import pandas as pd
import json
import os

name = 'Swimmer-v2_CMA_perturbations_64_state_renormalize_False_Sigma_1_best_64_test_1data'
with open('./Param/' + name + '.json') as file:
    dic = json.load(file)

if not os.path.exists("CSV_results"):
    os.mkdir("CSV_results")
df = pd.DataFrame(dic)
print(df['avg_fit'])
df.to_csv('./CSV_results/' + name + '.csv')