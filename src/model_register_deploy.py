# az acr credential show --name scoringprodu5091497c
# USERNAME              PASSWORD                          PASSWORD2
# --------------------  --------------------------------  --------------------------------
# scoringprodu5091497c  NnIrwJur5Fl=AY73i9gsXDtNtNnMPOEg  d84i1r+Q2VOtQ1iKVZB+qQLP1ixWQDra

import azureml.core
print('SDK version:', azureml.core.VERSION)

from azureml.core import Dataset
from azureml.core import Model
from azureml.core import Webservice
from azureml.core import Workspace
from azureml.exceptions import WebserviceException
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

from os import system as zsh
import os
parent_dir = os.path.realpath(__file__)
    # going up to parent directory of the current directory
for _ in range(2):
    parent_dir = os.path.dirname(parent_dir)
import json

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
            model = Model.register(**parameters)
        elif len(matched_models)>1:
            model_name = matched_models[0]['name']
            model = Model(workspace=ws, name = model_name)
        
    zsh('rm az_ml_model_list.json')    
    return model

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

model_path = os.path.join(parent_dir,'models')
registration_params = {
                       'workspace':ws,
                       'model_name':'sahulat-model',                # Name of the registered model in your workspace.
                       'model_path':model_path,  # Local file to upload and register as a model.
                    #    'sample_input_dataset':input_dataset,
                    #    'sample_output_dataset':output_dataset,
                       'resource_configuration':ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       'description':'Random Forest Classifier to predict credit default.',
                       'tags':{'area': 'lending', 'type': 'supervised classification'}
                    }
model = register_model(**registration_params)

print('Name:', model.name)
print('Version:', model.version)



service_name = 'sahulat-service'


try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies




myenv = Environment.get(workspace=ws, name="myenv")
from azureml.core.model import InferenceConfig


with open('src/score.py') as f:
    print(f.read())

inference_config = InferenceConfig(entry_script='src/score.py', environment=myenv)
aci_deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
from azureml.core.webservice import LocalWebservice

local_deployment_config = LocalWebservice.deploy_configuration(port=6789)

service = Model.deploy(
                       workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_deployment_config,
                       overwrite=True
                        )
service.wait_for_deployment(show_output=True)
print(service.get_logs())

