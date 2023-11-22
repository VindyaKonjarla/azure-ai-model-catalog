import os
import json
import concurrent.futures
import subprocess
from fetch_task import HfTask
from box import ConfigBox
#from model_inference_and_deployment import ModelInferenceAndDeployemnt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from azure.ai.ml import command
import mlflow
import json
import os
import sys
from box import ConfigBox
from mlflow.tracking.client import MlflowClient
from azureml.core import Workspace, Environment
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
from azure.ai.ml.entities import AmlCompute
import time
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
import ast


test_model_name = os.environ.get('test_model_name')
test_sku_type = os.environ.get('test_sku_type')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_queue = os.environ.get('test_queue')
test_set = os.environ.get('test_set')
test_keep_looping = os.environ.get('test_keep_looping')

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

def set_next_trigger_model(queue):
    print("In set_next_trigger_model...")
    model_list = list(queue.models)
    check_mlflow_model = test_model_name
    index = model_list.index(check_mlflow_model)
    print(f"index of {test_model_name} in queue: {index}")

    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print("::warning:: finishing the queue")
            next_model = ""

    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


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

def run_script(script):
    command = f"python {script}"
    return_code = os.system(command)
    return script, return_code

# def run_script(script_name):
#     subprocess.run(["python", script_name])

# def run_fine_tuning_task(task):
#     task_script_mapping = {
#         "text-classification": "FT_P_TC.py",
#         "question-answering": "FT_P_QA.py",
#         "token-classification": "FT_P_TC.py"
#         # Add more mappings as needed
#     }

#     script_name = task_script_mapping.get(task)
#     if script_name:
#         run_script(script_name)
#     else:
#         print(f"No script found for the fine-tune task: {task}")

# def run_fine_tuning_tasks(fine_tune_tasks):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(run_fine_tuning_task, task) for task in fine_tune_tasks]

#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Error running fine-tuning task: {e}")


# def run_fine_tuning_tasks(fine_tune_tasks):
#     task_script_mapping = {
#         "text-classification": "FT_P_TC.py",
#         "question-answering": "FT_P_QA.py",
#         "token-classification": "FT_P_NER.py",
#         "summarization": "FT_P_TS.py",
#         "translation": "FT_P_TT.py",
#         "text-generation": "FT_P_TG.py"
#     }

#     # scripts = task_script_mapping.get(task, [])
#     scripts = [task_script_mapping.get(task, "")for task in fine_tune_tasks]
#     scripts = [script for script in scripts if script]
#     if scripts:
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = [executor.submit(run_script, script) for script in scripts]

#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     script, return_code = result
#                     print(f"Script '{script}' completed with return code {return_code}")
#                 except Exception as e:
#                     print(f"Error running script '{script}': {e}")
#                     # sys.exit(1)
#     else:
#         print(f"No scripts found for the primary task: {fine_tune_tasks}")

# def run_fine_tuning_tasks(fine_tune_tasks):
#     task_script_mapping = {
#         "text-classification": "FT_P_TC.py",
#         "question-answering": "FT_P_QA.py",
#         "token-classification": "FT_P_NER.py",
#         "summarization": "FT_P_TS.py",
#         "translation": "FT_P_TT.py",
#         "text-generation": "FT_P_TG.py"
#     }

#     # scripts = task_script_mapping.get(task, [])
#     scripts = [task_script_mapping.get(task, "")for task in fine_tune_tasks]
#     scripts = [script for script in scripts if script]
    
#     # Add an error flag
#     error_occurred = False
    
#     if scripts:
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = [executor.submit(run_script, script) for script in scripts]

#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     script, return_code = result
#                     print(f"Script '{script}' completed with return code {return_code}")
#                     if return_code != 0:
#                         # Set the error flag to True if any script fails
#                         error_occurred = True
#                 except Exception as e:
#                     print(f"Error running script '{script}': {e}")
#                     # Set the error flag to True if any script fails
#                     error_occurred = True
    
#         # If an error occurred, print a message and exit with status 1
#         if error_occurred:
#             print("Error: At least one script failed.")
#             sys.exit(1)
#         else:
#             print("All scripts completed successfully.")
#     else:
#         print(f"No scripts found for the primary task: {fine_tune_tasks}")





def run_fine_tuning_tasks(fine_tune_tasks):
    task_script_mapping = {
        "text-classification": "FT_P_TC_HF.py",
        "question-answering": "FT_P_QA_HF.py",
        "token-classification": "FT_P_NER_HF.py",
        "summarization": "FT_P_TS_HF.py",
        "translation": "FT_P_TT_HF.py",
        "text-generation": "FT_P_TG_HF.py"
    }

    # scripts = task_script_mapping.get(task, [])
    scripts = [task_script_mapping.get(task, "")for task in fine_tune_tasks]
    scripts = [script for script in scripts if script]
    
    # Add an error flag
    error_occurred = False
    failed_scripts = []

    if scripts:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_script, script) for script in scripts]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    script, return_code = result
                    print(f"Script '{script}' completed with return code {return_code}")
                    if return_code != 0:
                        # Set the error flag to True if any script fails
                        error_occurred = True
                        failed_scripts.append(script)
                except Exception as e:
                    print(f"Error running script '{script}': {e}")
                    # Set the error flag to True if any script fails
                    error_occurred = True
                    failed_scripts.append(script)
    
        # If an error occurred, print a message with the names of failed scripts and exit with status 1
        if error_occurred:
            print("Error: The following scripts failed:")
            for failed_script in failed_scripts:
                print(f"- {failed_script}")
            sys.exit(1)
        else:
            print("All scripts completed successfully.")
    else:
        print(f"No scripts found for the primary task: {fine_tune_tasks}")



if __name__ == "__main__":
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        exit(1)

    queue = get_test_queue()

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)

    print(f"test_subscription_id: {queue['subscription']}")
    print(f"test_resource_group: {queue['resource_group']}")
    print(f"test_workspace_name: {queue['workspace']}")
    print(f"test_model_name: {test_model_name}")
    print(f"test_sku_type: {test_sku_type}")
    print(f"test_registry: {queue['registry']}")
    print(f"test_trigger_next_model: {test_trigger_next_model}")
    print(f"test_queue: {test_queue}")
    print(f"test_set: {test_set}")
    print("Here is my test model name: ", test_model_name)
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
    foundation_model = get_latest_model_version(workspace_ml_client, test_model_name.lower())

    primary_task = HfTask(model_name=test_model_name).get_task()
    print("Task is this: ", primary_task)
    # fine_tune_tasks = foundation_model.properties.get("finetune-recommended-sku", [])
    # fine_tune_tasks_str = foundation_model.properties.get("finetune-recommended-sku", "")
    # fine_tune_tasks = [task.strip() for task in fine_tune_tasks_str.split(",")] if fine_tune_tasks_str else []




    if primary_task:
        # Fetch fine-tune tasks for the specified model
        #fine_tune_tasks = foundation_model.properties.get("finetuning-tasks", [])
        fine_tune_tasks_str = foundation_model.properties.get("finetuning-tasks", "")
        fine_tune_tasks = [task.strip() for task in fine_tune_tasks_str.split(",")] if fine_tune_tasks_str else []

        print("1 finetune tasks from model card are:", {fine_tune_tasks_str})
        #print("2 finetune tasks from model card are:", {fine_tune_tasks})

        if fine_tune_tasks:
            # Run fine-tuning tasks in parallel
            run_fine_tuning_tasks(fine_tune_tasks)
        else:
            print(f"No fine-tune tasks found for the model: {model_name}")
    else:
        print(f"No primary task found for the model: {model_name}")

    # if primary_task:
    #     run_fine_tuning(primary_task)
    # else:
    #     print(f"No primary task found for the model: {model_name}")
    
