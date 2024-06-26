from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from azure.ai.ml import command
import mlflow
import json
import os
import sys
from box import ConfigBox
from azure.ai.ml.entities import Model
from mlflow.tracking.client import MlflowClient
from azureml.core import Workspace, Environment
from azure.ai.ml import MLClient
from azureml.core import Model
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute
import time
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
import ast
import re
from datetime import datetime
import time
from azureml.core import Model
from azure.ai.ml.entities import Model

check_override = True


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


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))

# this is useful if you want to force a specific sku for a model


def get_sku_override():
    try:
        with open('../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None




def run_azure_ml_job(code, command_to_run, environment, compute, environment_variables):
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



def get_latest_model_version(workspace_ml_client, test_model_name):
    print("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(test_model_name))
    
    if len(version_list) == 0:
        print("Model not found in registry")
        foundation_model_name = None  # Set to None if the model is not found
        foundation_model_id = None  # Set id to None as well
    else:
        model_version = version_list[0].version
        foundation_model = workspace_ml_client.models.get(
            test_model_name, model_version)
        print(
            "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                foundation_model.name, foundation_model.version, foundation_model.id
            )
        )
        foundation_model_name = foundation_model.name  # Assign the value to a new variable
        foundation_model_id = foundation_model.id  # Assign the id to a new variable
    
    # Check if foundation_model_name and foundation_model_id are None or have values
    if foundation_model_name and foundation_model_id:
        print(f"Latest model {foundation_model_name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        print("foundation_model.name:", foundation_model_name)
        print("foundation_model.id:", foundation_model_id)
    else:
        print("No model found in the registry.")
    
    #print(f"Model Config : {latest_model.config}")
    return foundation_model

# def get_latest_model_version_ft(registry_ml_client_sku, test_model_name, version_to_fetch):
#     print("In get_latest_model_version...")
#     version_list = list(registry_ml_client_sku.models.list(test_model_name))
    
#     if len(version_list) == 0:
#         print("Model not found in registry")
#         foundation_model_ft_name = None  # Set to None if the model is not found
#         foundation_model_ft_id = None  # Set id to None as well
#     else:
#         model_version = version_list[0].version
#         foundation_model_ft = registry_ml_client_sku.models.get(
#             test_model_name, model_version)
#         print(
#             "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
#                 foundation_model_ft.name, foundation_model_ft.version, foundation_model_ft.id
#             )
#         )
#         foundation_model_ft_name = foundation_model_ft.name  # Assign the value to a new variable
#         foundation_model_ft_id = foundation_model_ft.id  # Assign the id to a new variable
    
#     # Check if foundation_model_name and foundation_model_id are None or have values
#     if foundation_model_ft_name and foundation_model_ft_id:
#         print(f"Latest model {foundation_model_ft_name} version {foundation_model_ft.version} created at {foundation_model_ft.creation_context.created_at}")
#         print("foundation_model.name:", foundation_model_ft_name)
#         print("foundation_model.id:", foundation_model_ft_id)
#     else:
#         print("No model found in the registry.")
    
#     #print(f"Model Config : {latest_model.config}")
#     return foundation_model_ft

def get_latest_model_version_ft(registry_ml_client_sku, test_model_name, version_to_fetch):
    print("In get_latest_model_version...")
    version_list = list(registry_ml_client_sku.models.list(test_model_name))
    
    if len(version_list) == 0:
        print("Model not found in registry")
        foundation_model_ft_name = None  # Set to None if the model is not found
        foundation_model_ft_id = None  # Set id to None as well
    else:
        # Find the specified version in the list
        for model_version in version_list:
            if model_version.version == version_to_fetch:
                foundation_model_ft = registry_ml_client_sku.models.get(test_model_name, version_to_fetch)
                break
        else:
            # If the specified version is not found, use the latest version
            foundation_model_ft = registry_ml_client_sku.models.get(test_model_name, version_list[0].version)

        print(
            "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                foundation_model_ft.name, foundation_model_ft.version, foundation_model_ft.id
            )
        )
        foundation_model_ft_name = foundation_model_ft.name  # Assign the value to a new variable
        foundation_model_ft_id = foundation_model_ft.id  # Assign the id to a new variable
    
    # Check if foundation_model_name and foundation_model_id are None or have values
    if foundation_model_ft_name and foundation_model_ft_id:
        print(f"Latest model {foundation_model_ft_name} version {foundation_model_ft.version} created at {foundation_model_ft.creation_context.created_at}")
        print("foundation_model.name:", foundation_model_ft_name)
        print("foundation_model.id:", foundation_model_ft_id)
    else:
        print("No model found in the registry.")
    
    #print(f"Model Config : {latest_model.config}")
    return foundation_model_ft



def get_training_and_optimization_parameters(foundation_model):
    # Training parameters
    training_parameters = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 2e-5,
        "metric_for_best_model": "f1_macro",
    }
    print(f"The following training parameters are enabled - {training_parameters}")

    # Optimization parameters
    if "model_specific_defaults" in foundation_model.tags:
        optimization_parameters = ast.literal_eval(foundation_model.tags["model_specific_defaults"])
    else:
        optimization_parameters = {
            "apply_lora": "true",
            "apply_deepspeed": "true",
            "apply_ort": "true",
        }
    print(f"The following optimizations are enabled - {optimization_parameters}")

    return training_parameters, optimization_parameters



def get_model_version_from_json(test_model_name):
    with open('models_versions.json', 'r') as json_file:
        models_versions = json.load(json_file)
        return models_versions.get(test_model_name, None)


def create_or_get_aml_compute(workspace_ml_client, compute_cluster, compute_cluster_size):
    try:
        compute = workspace_ml_client.compute.get(compute_cluster)
        print(f"The compute cluster '{compute_cluster}' already exists! Reusing it for the current run")
    except Exception as ex:
        print(f"Looks like the compute cluster '{compute_cluster}' doesn't exist. Creating a new one with compute size '{compute_cluster_size}'!")

        try:
            # Attempt to create a dedicated compute
            print("Attempt #1 - Trying to create a dedicated compute")
            tier = "Dedicated"
        except Exception as e:
            print(e)
            raise

        # Provision the compute
        compute = AmlCompute(
            name=compute_cluster,
            size=compute_cluster_size,
            tier=tier,
            max_instances=2,  # For multi-node training, set this to an integer value more than 1
        )
        workspace_ml_client.compute.begin_create_or_update(compute).wait()

    # Sanity check on the created compute
    compute = workspace_ml_client.compute.get(compute_cluster)

    if compute.provisioning_state.lower() == "failed":
        raise ValueError(f"Provisioning failed. Compute '{compute_cluster}' is in a failed state. Please try creating a different compute.")

    # Determine the number of GPUs in a single node of the selected 'compute_cluster_size' compute
    gpu_count_found = False
    workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
    available_sku_sizes = []

    for compute_sku in workspace_compute_sku_list:
        available_sku_sizes.append(compute_sku.name)
        if compute_sku.name.lower() == compute.size.lower():
            gpus_per_node = compute_sku.gpus
            gpu_count_found = True

    # If the GPU count is not found, print an error
    if gpu_count_found:
        print(f"Number of GPUs in compute '{compute_cluster}': {gpus_per_node}")
    else:
        raise ValueError(f"Number of GPUs in compute '{compute_cluster}' not found. Available skus are: {available_sku_sizes}. This should not happen. Please check the selected compute cluster: {compute_cluster} and try again.")
    
    return compute, gpus_per_node, compute_cluster




def create_and_run_azure_ml_pipeline(
    foundation_model,
    compute_cluster,
    gpus_per_node,
    training_parameters,
    optimization_parameters,
    experiment_name,
):
    # Fetch the pipeline component
    pipeline_component_func = registry_ml_client.components.get(
        name="text_classification_pipeline", label="latest"
    )
    print("Pipeline Component ID:", pipeline_component_func.id)


    # Model registration function
    #timestamp = str(time.time())
   

    # def register_model_to_workspace(
    #     workspace_ml_client, pipeline_job, test_model_name, timestamp
    # ):
    #     print("Registering the model inside loop...")
    #     model_path_from_job = "azureml://jobs/{0}/outputs/{1}".format(
    #         pipeline_job.name, "trained_model"
    #     )
    #     finetuned_model_name = (
    #         "FT-TC-" + str(test_model_name) + "-oss"
    #     )
    #     finetuned_model_name = finetuned_model_name.replace("/", "-")
    #     print("The Finetuned model name inside loop:", finetuned_model_name)

    #     print("Path to register model inside loop: ", model_path_from_job)

    #     prepare_to_register_model = Model(
    #     path=model_path_from_job,
    #     type=AssetTypes.MLFLOW_MODEL,
    #     name=finetuned_model_name,
    #     version=timestamp,  # use timestamp as version to avoid version conflict
    #     description= test_model_name + " fine-tuned model to text-classification",
    #     )
    #     print("prepare to register model inside loop:", prepare_to_register_model)
    
    #     registered_model = workspace_ml_client.models.create_or_update(
    #         prepare_to_register_model
    #     )
    
    #     print("Registered model inside loop: \n", registered_model)

    # Define the pipeline job
    @pipeline()
    def create_pipeline():
        text_classification_pipeline = pipeline_component_func(
            mlflow_model_path=foundation_model.id,
            compute_model_import=compute_cluster,
            compute_preprocess=compute_cluster,
            compute_finetune=compute_cluster,
            compute_model_evaluation=compute_cluster,
            train_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_train.jsonl"
            ),
            validation_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_validation.jsonl"
            ),
            test_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_test.jsonl"
            ),
            evaluation_config=Input(
                type="uri_file",
                path="./emotion-dataset/text-classification-config.json",
            ),
            sentence1_key="text",
            label_key="label_string",
            number_of_gpu_to_use_finetuning=gpus_per_node,
            **training_parameters,
            **optimization_parameters
        )
        return {"trained_model": text_classification_pipeline.outputs.mlflow_model_folder}

    # Create the pipeline object
    pipeline_object = create_pipeline()

    # Configure pipeline settings
    pipeline_object.settings.force_rerun = True
    pipeline_object.settings.continue_on_step_failure = False

    # Submit the pipeline job
    pipeline_job = workspace_ml_client.jobs.create_or_update(
        pipeline_object, experiment_name=experiment_name
    )

    workspace_ml_client.jobs.stream(pipeline_job.name)
    print("Pipeline Job Completed")

    # # Call the model registration function
    # register_model_to_workspace(
    #     workspace_ml_client, pipeline_job, test_model_name, timestamp
    # )

    return pipeline_job


if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()


    # if test_trigger_next_model == "true":
        # set_next_trigger_model(queue)
    # print values of all above variables
    print ("Running for Text Classification")
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: {queue['registry']}")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")
    print("Here is my test model name : ", test_model_name)
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    print("workspace_name : ", queue.workspace)
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
    registry_ml_client_sku = MLClient(credential, registry_name="azureml")
    expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                            "$", "%", "^", "&", "*", "<", ">", "?", "!", "~"]
    # Create the regular expression to ignore
    regx_for_expression = re.compile(
        '|'.join(map(re.escape, expression_to_ignore)))
    # Check the model_name contains any of there character
    expression_check = re.findall(regx_for_expression, test_model_name)
    if expression_check:
        # Replace the expression with hyphen
        test_model_name  = regx_for_expression.sub("-", test_model_name)
    print("model name replaced with - :", {test_model_name})

    version_to_fetch = get_model_version_from_json(test_model_name.lower())
    
    if version_to_fetch is None:
        print(f"Error: Model version for {test_model_name} not found in the JSON file.")

    
    foundation_model_ft = get_latest_model_version_ft(registry_ml_client_sku, test_model_name.lower(), version_to_fetch)
    # foundation_model_ft = get_latest_model_version_ft(registry_ml_client_sku, test_model_name.lower())
    fine_tune_sku = foundation_model_ft.properties.get("finetune-recommended-sku")
    print("Finetune-recommended-sku:", {fine_tune_sku})
    registry_ml_client = MLClient(credential, registry_name="azureml-preview-test1")
    # experiment_name = "text-classification-emotion-detection"
    experiment_name = "oss-text-classification-"+ test_model_name
    print("Experiment name is:", {experiment_name})

    
    compute_cluster_size = fine_tune_sku
    compute_cluster = "donotdelete-" + compute_cluster_size.replace('_', '-')
    print("Modified compute_cluster_size:", compute_cluster_size)
    print("Modified compute_cluster_size:", {compute_cluster})


    
    # Call the function
    compute, gpus_per_node, compute_cluster = create_or_get_aml_compute(workspace_ml_client, compute_cluster, compute_cluster_size)

    
    # compute = create_or_get_compute_target(workspace_ml_client, queue.compute)
    print("printing:",{compute})
    env_list = workspace_ml_client.environments.list(name=queue.environment)
    latest_version = 0
    for env in env_list:
        if latest_version <= int(env.version):
            latest_version = int(env.version)
    print("Latest Environment Version:", latest_version)
    latest_env = workspace_ml_client.environments.get(
        name=queue.environment, version=str(latest_version))
    print("Latest Environment :", latest_env)
    expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                            "$", "%", "^", "&", "*", "<", ">", "?", "!", "~"]
    # Create the regular expression to ignore
    regx_for_expression = re.compile(
        '|'.join(map(re.escape, expression_to_ignore)))
    # Check the model_name contains any of there character
    expression_check = re.findall(regx_for_expression, test_model_name)
    if expression_check:
        # Replace the expression with hyphen
        test_model_name  = regx_for_expression.sub("-", test_model_name)

    print("model name replaced with - :", {test_model_name})
    version_list = list(workspace_ml_client.models.list(test_model_name))

    client = MlflowClient()
    
    #foundation_model, foundation_model_name = get_latest_model_version(workspace_ml_client, test_model_name.lower())
    foundation_model = get_latest_model_version(workspace_ml_client, test_model_name.lower())
    training_parameters, optimization_parameters = get_training_and_optimization_parameters(foundation_model)
    #gpus_per_node = find_gpus_in_compute(workspace_ml_client, compute)
    print(f"Number of GPUs in compute: {gpus_per_node}")

    timestamp_str = str(time.time())
    timestamp = timestamp_str.split(".")[0]
    print("timestamp_str:",{timestamp_str})
    print("timestamp:",{timestamp})
    
    


    try:
        pipeline_job = create_and_run_azure_ml_pipeline(
            foundation_model, compute_cluster, gpus_per_node, training_parameters, optimization_parameters, experiment_name
        )
        print("pipeline job completed")

    except Exception as e:
        # If an exception occurs, print the error message and exit with a non-zero exit code
        print(f"Error running Azure ML Pipeline: {str(e)}")
        sys.exit(1) 

    print("Finetuned and the registered model for Text-classification successfully")
