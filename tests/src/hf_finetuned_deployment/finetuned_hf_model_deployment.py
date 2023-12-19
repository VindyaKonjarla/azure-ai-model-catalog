from azureml.core import Workspace
from model_inference_and_deployment import ModelInferenceAndDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import MLClient
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
from fetch_task import HfTask
from azure.core.exceptions import ResourceNotFoundError
from fetch_model_detail import ModelDetail
from azure.core.exceptions import (
    ResourceNotFoundError
)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint
)
import re
#from azure.ai.ml.entities import Model

logger = get_logger(__name__)

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

actual_model_name = os.environ.get('actual_model_name')

azure_ml_model_name = os.environ.get('azure_ml_model_name')

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

def create_or_get_compute_target(ml_client, compute, instance_type):
    cpu_compute_target = compute
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except ResourceNotFoundError:
        logger.info("Creating a new compute...")
        compute = AmlCompute(
            name=cpu_compute_target, size=instance_type, idle_time_before_scale_down=120, min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()

    return compute

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
    index = model_list.index(test_model_name)
    #model_name_without_slash = test_model_name.replace('/', '-')
    # check_mlflow_model = "MLFlow-"+test_model_name
    # import_alias_model_name = f"MLFlow-Import-{test_model_name}"
    # mp_alias_model_name = f"MLFlow-MP-{test_model_name}"

    # if check_mlflow_model in model_list:
    #     index = model_list.index(check_mlflow_model)
    # elif import_alias_model_name in model_list:
    #     index = model_list.index(import_alias_model_name)
    # elif mp_alias_model_name in model_list:
    #     index = model_list.index(mp_alias_model_name)
    # else:
    #     index = model_list.index("MLFlow-Evaluate-"+test_model_name)

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

def create_endpoint(workspace_ml_client, endpoint_name):
    try:
        logger.info("Inside creating the endpoint method")
        endpoint = workspace_ml_client.online_endpoints.get(name=endpoint_name)
        return endpoint
    except ResourceNotFoundError as e:
        logger.warning("The endpoint do not exist and now creting new endpoint")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key"
        )
        logger.warning("update the endpoint in the workspace")
        workspace_ml_client.online_endpoints.begin_create_or_update(
            endpoint).wait()
        return endpoint
    except Exception as e:
        logger.error(f"Failed due to this : {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    # print values of all above variables
    logger.info (f"test_subscription_id: {queue['subscription']}")
    logger.info (f"test_resource_group: {queue['subscription']}")
    logger.info (f"test_workspace_name: {queue['workspace']}")
    logger.info (f"test_model_name: {test_model_name}")
    logger.info (f"test_sku_type: {test_sku_type}")
    logger.info (f"test_trigger_next_model: {test_trigger_next_model}")
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
    # try:
    #     workspace_ml_client = MLClient.from_config(credential=credential)
    # except:
    #     workspace_ml_client = MLClient(
    #         credential=credential,
    #         subscription_id=queue.subscription,
    #         resource_group_name=queue.resource_group,
    #         workspace_name=queue.workspace
    #     )
    workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    logger.info(f"work_space_ml_client : {workspace_ml_client}")
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    # registry_ml_client = MLClient(
    #     credential=credential,
    #     registry_name=queue.registry
    # )
    # task = HfTask(model_name=test_model_name).get_task()
    # logger.info(f"Task is this : {task} for the model : {test_model_name}")
    
    #Fetch model from workspace
    # model_detail = ModelDetail(workspace_ml_client=workspace_ml_client)
    # registered_model = model_detail.get_model_detail(
    #     test_model_name=test_model_name)
    #task = registered_model.flavors['hftransformersv2']['task_type']
    # Connect to registry
    azureml_registry = MLClient(credential, registry_name="azureml")
    # Fetch model form the registry
    model_detail = ModelDetail(workspace_ml_client=azureml_registry)
    foundation_model = model_detail.get_model_detail(
        test_model_name=azure_ml_model_name)
    recomended_sku_list = foundation_model.properties.get("inference-recommended-sku", None)
    if recomended_sku_list != None:
        instance_type = list(recomended_sku_list.split(','))[0]
        logger.info(f"Recomended SKU type is this one {instance_type}")
    else:
        recomended_sku_list = foundation_model.tags.get("inference_compute_allow_list", None)
        if recomended_sku_list != None:
            exp_tobe_replaced = ["[", "]", "'"]
            regx_for_expression = re.compile('|'.join(map(re.escape, exp_tobe_replaced)))
            recomended_sku_list = re.sub(regx_for_expression, "", recomended_sku_list)
            instance_type = recomended_sku_list.split(',')[0]
            logger.info(f"Recomended SKU type is this one {instance_type}")
        else:
            logger.info("Deployment task not supported here")
            sys.exit(1)
    
    compute = instance_type.replace("_", "-")
    logger.info(f"instance : {instance_type} and compute is : {compute}")
    
    compute_target = create_or_get_compute_target(
        ml_client=workspace_ml_client, compute=compute, instance_type=instance_type)
    #endpoint_name = queue.workspace.split("-")[-1] + "-" + compute.lower()
    endpoint_name = compute.lower()
    endpoint = create_endpoint(
        workspace_ml_client=workspace_ml_client,
        endpoint_name=endpoint_name
    )   
    logger.info("Proceeding with inference and deployment")
    
    task_file_loc = f"task_short_name.json"
    f = open(task_file_loc)
    task_shortcut = ConfigBox(json.load(f))
    # with open(task_file_loc) as f:
    #     ConfigBox(json.load(f))
    tasks = task_shortcut.get(azure_ml_model_name, None)
    if tasks != None:
        for task, starting_name in tasks.items():
            #starting_name = task_shortcut.get(task, None)
            if starting_name != None:
                fianl_model_name = f"{starting_name}-{test_model_name}"
                logger.info(f"Final model name needs to be found is {fianl_model_name}")
                try:
                    model_detail = ModelDetail(workspace_ml_client=workspace_ml_client)
                    registered_model = model_detail.get_model_detail(
                            test_model_name=fianl_model_name)
                    logger.info(f"The registered model is this : {registered_model}")
                    InferenceAndDeployment = ModelInferenceAndDeployemnt(
                        test_model_name=test_model_name,
                        workspace_ml_client=workspace_ml_client
                    )
                    InferenceAndDeployment.model_infernce_and_deployment(
                        instance_type=instance_type,
                        task=task,
                        latest_model=registered_model,
                        #compute=compute,
                        endpoint=endpoint,
                        actual_model_name=actual_model_name
                    )
                except ResourceNotFoundError:
                    #logger.warning("Model Resource Not found in the registry")
                    logger.warning(f"The model not found in the workspace {fianl_model_name}")

    # InferenceAndDeployment = ModelInferenceAndDeployemnt(
    #     test_model_name=test_model_name,
    #     workspace_ml_client=workspace_ml_client
    # )
    # InferenceAndDeployment.model_infernce_and_deployment(
    #     instance_type=instance_type,
    #     task=task,
    #     latest_model=registered_model,
    #     #compute=compute,
    #     endpoint=endpoint,
    #     actual_model_name=actual_model_name
    # )