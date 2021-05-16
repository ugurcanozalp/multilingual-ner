#https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python
#https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/deploy-to-cloud

from azureml.core import Workspace
ws = Workspace(subscription_id="<subscription_id>",
               resource_group="<resource_group>",
               workspace_name="<workspace_name>")


import urllib.request
from azureml.core.model import Model

# Register model
model = Model.register(ws, model_name="multiner_model", model_path="ner_models/az")

from azureml.core import Environment
from azureml.core.model import InferenceConfig

env = Environment(name="multiner_environment")
dummy_inference_config = InferenceConfig(
    environment=env,
    source_directory="./source_dir",
    entry_script="./echo_score.py",
)

from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port=6789)

service = Model.deploy(
    ws,
    "myservice",
    [model],
    dummy_inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)

print(service.get_logs())

import requests
import json

uri = service.scoring_uri
requests.get("http://localhost:6789")
headers = {"Content-Type": "application/json"}
data = {
	"text": """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """,
}
data = json.dumps(data)
response = requests.post(uri, data=data, headers=headers)
print(response.json())

##############################################################
env = Environment(name='myenv')
python_packages = ['nltk', 'numpy', 'onnxruntime']
for package in python_packages:
    env.python.conda_dependencies.add_pip_package(package)

inf_config = InferenceConfig(environment=env, source_directory='./source_dir', entry_script='./score.py')

service = Model.deploy(
    ws,
    "myservice",
    [model],
    inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)

print(service.get_logs())