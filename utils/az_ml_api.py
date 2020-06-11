from os import system as zsh
import json
zsh('az ml model list --output json > az_ml_model_list.json')

with open('az_ml_model_list.json','rb') as file:
	az_ml_list = json.load(file)

import pandas as pd

df = pd.read_json('az_ml_model_list.json')
ids = list(set(df['id']))
for id in ids:
   zsh('az ml model delete --model-id  '+str(id))
zsh('rm az_ml_model_list.json')
