from azureml.core import Workspace, Environment
from model_inference_and_deployment import ModelInferenceAndDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
#from HF_credentials import get_huggingface_token
huggingface_token = os.environ.get("HF_TOKEN")

# constants
check_override = True

logger = get_logger(__name__)

def get_huggingface_token():
    return os.environ.get('HF_token', '')

def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../../config/errors.json') as f:
        return json.load(f)


error_messages = get_error_messages()
# model to test
test_model_name = os.environ.get('test_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# bool to decide if we want to trigger the next model in the queue
test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue,
# which means that the first model in the queue is triggered again after the last model is tested
test_keep_looping = os.environ.get('test_keep_looping')

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model


def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None


# finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    logger.info("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    check_mlflow_model = test_model_name.replace('/', '-')
    if check_mlflow_model in model_list:
        index = model_list.index(check_mlflow_model)
    else:
        index = model_list.index(test_model_name)
    #index = model_list.index(test_model_name)
    logger.info(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            logger.warning("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        logger.info(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def create_or_get_compute_target(ml_client,  compute):
    cpu_compute_target = compute
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        logger.info("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size=compute, min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()

    return compute


def run_azure_ml_job(code, command_to_run, environment, compute, environment_variables):
    logger.info("Creating the command object method")
    command_job = command(
        code=code,
        command=command_to_run,
        environment=environment,
        compute=compute,
        environment_variables=environment_variables
    )
    return command_job


def create_and_get_job_studio_url(command_job, workspace_ml_client):

    #ml_client = mlflow.tracking.MlflowClient()
    returned_job = workspace_ml_client.jobs.create_or_update(command_job)
    # wait for the job to complete
    workspace_ml_client.jobs.stream(returned_job.name)
    return returned_job.studio_url
def run_model(test_model_name, queue):
    try:
        # Code to run the model (using generic_model_download_and_register1.py)
        command_job = run_azure_ml_job(
            code="./",
            command_to_run="python generic_model_download_and_register1.py",
            environment=latest_env,
            compute=queue.compute,
            environment_variables={
                "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT": 600.0,
                "test_model_name": test_model_name
            }
        )
        create_and_get_job_studio_url(command_job, workspace_ml_client)
 
        InferenceAndDeployment = ModelInferenceAndDeployemnt(
            test_model_name=test_model_name.lower(),
            workspace_ml_client=workspace_ml_client,
            registry=queue.registry
        )
        InferenceAndDeployment.model_infernce_and_deployment(
            instance_type=queue.instance_type
        )
 
    except Exception as e:
        print(f"Model {test_model_name} failed with error: {e}")
        return False  # Indicate model failure
 
    return True  # Indicate model success
 
def run_all_models(queue, model_list):
    success = False
    for test_model_name in model_list:
        success = run_model(test_model_name, queue)
        if success:
            break  # Exit loop if a model runs successfully
 
    return success
 
if __name__ == "__main__":
    queue = get_test_queue()
    model_list = list(queue.models)
 
    # Run all models in the list
    success = run_all_models(queue, model_list)
 
    # If no models were successful, print a message
    if not success:
        print("All models in the queue failed.")
    # Trigger next model if needed
    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
