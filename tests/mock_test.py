import pandas
import pickle
import pandas as pd
from train_forest import global_train_parameters
from sklearn.ensemble import RandomForestClassifier
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import LocalWebservice
from azureml.core.webservice import AciWebservice
from azureml.core import Workspace
from azureml.core.environment import CondaDependencies
import requests
import json
# from register_model_deploy_local import register_model
from azureml.core import Workspace
from os import system as zsh
def register_model(**parameters):
    zsh('az ml model list --output json > az_ml_model_list.json')
    model_name = parameters['model_name']
    # model_id = parameters['id']
    # model_version = parameters['version']
    ws = parameters['workspace']
    with open('az_ml_model_list.json') as models_json:
        models = json.load(models_json)
        matched_models = [model for model in models if model['name']==model_name]
        if len(matched_models)==1:
            print('Found the model\n')
            model = Model(workspace=ws, name = model_name)
        elif len(matched_models)==0:
            print('Provided model {} has not been found\n'.format(model_name))
            print('registering new model in Azure ...\n')
            model = Model.register(**model_registration_parameters)
        elif len(matched_models)>1:
            model_name = matched_models[0]['name']
            model = Model(workspace=ws, name = model_name)
        
    zsh('rm az_ml_model_list.json')    
    return model

# prepare dummy model
target_column = global_train_parameters['target_column']
processed_data_path = global_train_parameters['processed_data_path']
train_data = pd.read_csv(processed_data_path)
Y_train = train_data.pop(target_column)
X_train = train_data
model = RandomForestClassifier()
model.fit(X_train, Y_train)
filename = 'dummy.pkl'
pickle.dump(model, open(filename, 'wb'))

# register the dummy model
ws = Workspace.from_config()
registration_params = {
                       'model_path':"dummy.pkl",
                       'model_name' : "dummy-model",
                       'description' : "mock test deployment",
                       'workspace' : ws
}
model = register_model(**registration_params)

myenv = Environment(name='my_env')
myenv.get(workspace=ws, name='ls-ds-ml-env')
conda_dep = CondaDependencies()
conda_dep.add_pip_package("azureml-defaults==1.3.0")
conda_dep.add_pip_package("joblib")
conda_dep.add_pip_package("json")
myenv.python.conda_dependencies=conda_dep
inference_config = InferenceConfig(entry_script="src/dummy_score.py", environment=myenv)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_name = 'dummy-service'
try:
   service = Model.deploy(
    ws, service_name, [model], inference_config, deployment_config=aci_config)
   service.wait_for_deployment(True)
   logs = service.get_logs()
   print(logs)
except:
   print('Exception occured !\n')
# Check its state
state = service.state
print(state)

if state.startswith('Unhealthy'):
	print(service.get_logs())

print(service.endpoint)
