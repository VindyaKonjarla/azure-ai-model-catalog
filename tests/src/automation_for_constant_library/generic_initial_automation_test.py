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

logger = get_logger(__name__)

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# model to test
test_model_name = os.environ.get('test_model_name')

def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
    
    
def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None
    
if __name__ == "__main__":
        # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None :
        logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set ")
        exit(1)

    queue = get_test_queue()
    
    
# print values of all above variables
    logger.info (f"test_subscription_id: {queue['subscription']}")
    logger.info (f"test_resource_group: {queue['subscription']}")
    logger.info (f"test_workspace_name: {queue['workspace']}")
    logger.info (f"test_model_name: {test_model_name}")
    logger.info (f"test_sku_type: {test_sku_type}")
    #logger.info (f"test_trigger_next_model: {test_trigger_next_model}")
    logger.info (f"test_queue: {test_queue}")
    logger.info (f"test_set: {test_set}")
    logger.info(f"Here is my test model name : {test_model_name}")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    logger.info(f"workspace_name : {queue.workspace}")
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    # compute_target = create_or_get_compute_target(
    #     workspace_ml_client, queue.compute)
    # environment_variables = {"AZUREML_ARTIFACTS_DEFAULT_TIMEOUT":600.0,"test_model_name": test_model_name}
    # env_list = workspace_ml_client.environments.list(name=queue.environment)
    # latest_version = 0
    # for env in env_list:
    #     if latest_version <= int(env.version):
    #         latest_version = int(env.version)
    # logger.info(f"Latest Environment Version: {latest_version}")
    # latest_env = workspace_ml_client.environments.get(
    #     name=queue.environment, version=str(latest_version))
    # logger.info(f"Latest Environment : {latest_env}")
    # command_job = run_azure_ml_job(code="./", command_to_run="python generic_model_download_and_register.py",
    #                                environment=latest_env, compute=queue.compute, environment_variables=environment_variables)
    # create_and_get_job_studio_url(command_job, workspace_ml_client)

    InferenceAndDeployment = ModelInferenceAndDeployemnt(
        test_model_name=test_model_name.lower(),
        workspace_ml_client=workspace_ml_client,
        registry=queue.registry
    )
    InferenceAndDeployment.model_infernce_and_deployment(
        instance_type=queue.instance_type
    )
