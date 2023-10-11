from transformers import AutoModel, AutoTokenizer, AutoConfig
import transformers
from urllib.request import urlopen
from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.dsl import pipeline
from huggingface_hub import HfApi
import re
import pandas as pd
import os
import shutil
import json
import os
import sys
from box import ConfigBox
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError
# from utils.logging import get_logger
test_model_name = os.environ.get('test_model_name')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_keep_looping = os.environ.get('test_keep_looping')
test_queue = os.environ.get('test_queue')
test_set = os.environ.get('test_set')
experiment_name=os.environ.get('experiment_name')
compute_name = "model-import-cluster"
# COMPUTE=os.environ.get('COMPUTE')
# experiment_name = f"Import Model Pipeline"
URL = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=10000"
COLUMNS_TO_READ = ["modelId", "pipeline_tag", "tags"]
LIST_OF_COLUMNS = ['modelId', 'downloads',
       'lastModified', 'tags', 'pipeline_tag']
TASK_NAME = ['fill-mask', 'token-classification', 'question-answering',
     'summarization', 'text-generation', 'text-classification', 'translation']
STRING_TO_CHECK = 'transformers'
FILE_NAME = "task_and_library.json"

update_existing_model=True
Reg_Model=test_model_name.replace('/','-')
huggingface_model_exists_in_registry = False
def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
def get_task() -> str:
    hf_api = HfApi()
    # Get all the1 models in the list
    models = hf_api.list_models(
        full=True, sort='lastModified', direction=-1)
    # Unpack all values from the generator object
    required_data = [i for i in models]

    daata_dict = {}
    # Loop through the list
    for data in required_data:
        # Loop through all the column present in the list
        for key in data.__dict__.keys():
            if key in LIST_OF_COLUMNS:
                # Check the dictionary already contains a value for that particular column
                if daata_dict.get(key) is None:
                    # If the column and its value is not present then insert column and an empty list pair to the dictionary
                    daata_dict[key] = []
                # Get the value for that particular column
                values = daata_dict.get(key)
                if key == 'tags':
                    # If its tag column extract value if it is nonne then bydefault return a list with string Empty
                    values.append(data.__dict__.get(key, ["Empty"]))
                else:
                    values.append(data.__dict__.get(key, "Empty"))
                daata_dict[key] = values
    # Convert dictionary to the dataframe
    df = pd.DataFrame(daata_dict)
    # Find the data with the model which will be having trasnfomer tag
    df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]
    # Retrive the data whose task is in the list
    df = df[df['pipeline_tag'].isin(TASK_NAME)]

    # Find the data with that particular name
    required_data = df[df.modelId.apply(lambda x: x ==  test_model_name)]
    # Get the task
    required_data = required_data["pipeline_tag"].to_string()
    # Create pattern fiel number and space
    pattern = r'[0-9\s+]'
    # Replace number and space
    final_data = re.sub(pattern, '', required_data)
    return final_data
def set_next_trigger_model(queue):
    print("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    # check_mlflow_model = "MLFlow-Import-"+test_model_name
    
    check_mlflow_model = "IM-Flavourcheck-"+test_model_name
    index = model_list.index(check_mlflow_model)
    #index = model_list.index(test_model_name)
    print(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)

@pipeline
def model_import_pipeline(model_id,compute_name,update_existing_model, task_name):

    import_model_job = import_model(model_id=test_model_name,compute=compute_name, task_name=task_name,update_existing_model=update_existing_model)
    # Set job to not continue on failure
    import_model_job.settings.continue_on_step_failure = False
    return {"model_registration_details": import_model_job.outputs.model_registration_details}
if __name__ == "__main__":
    
    TASK_NAME = get_task()
    print("TASK_NAME:==",TASK_NAME)
    print("test_queue:==",test_queue)
    queue = get_test_queue()
    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
        # print("workspace_name : ", queue.workspace)
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ml_client_registry = MLClient(credential, registry_name=queue.registry)
    registry_mlclient = MLClient(credential, registry_name="azureml")
    version_list = list(registry_mlclient.models.list(Reg_Model))
    if len(version_list) == 0:
        print("Model not found in registry")
    else:
        model_version = version_list[0].version
        foundation_model = registry_mlclient.models.get(
        Reg_Model, model_version)
        print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
            foundation_model.name, foundation_model.version, foundation_model.id))
    # computelist=foundation_model.properties.get("inference-recommended-sku", "Standard_E16s_v3")
    # a = computelist.index(',')
    # # COMPUTE = computelist[:a]
    # # COMPUTE="Standard-E64s-v3"
    # try:
    #     _ = workspace_ml_client.compute.get(compute_name)
    #     print("Found existing compute target.")
    # except ResourceNotFoundError:
    #     print("Creating a new compute target...")
    #     compute_config = AmlCompute(
    #         name=compute_name,
    #         type="amlcompute",
    #         size="STANDARD_E64S_V3",
    #         idle_time_before_scale_down=120,
    #         min_instances=0,
    #         max_instances=6,
    #     )
    #     workspace_ml_client.begin_create_or_update(compute_config).result()
    # import_model = ml_client_registry.components.get(name="import_model_oss_test", label="latest")
    # pipeline_object = model_import_pipeline(
    #     model_id=test_model_name,
    #     compute_name=compute_name,
    #     task_name=TASK_NAME,
    #     update_existing_model=update_existing_model,
        
    # )
    # pipeline_object.identity = UserIdentityConfiguration()

    # pipeline_object.settings.force_rerun = True


    # pipeline_object.settings.default_compute = compute_name
    # schedule_huggingface_model_import = (
    #     not huggingface_model_exists_in_registry
    #     and test_model_name not in [None, "None"]
    #     and len(test_model_name) > 1
    # )
    # print(
    #     f"Need to schedule run for importing {test_model_name}: {schedule_huggingface_model_import}")

    # huggingface_pipeline_job = None
    # # if schedule_huggingface_model_import:
    #     # submit the pipeline job
    # huggingface_pipeline_job = workspace_ml_client.jobs.create_or_update(
    #     pipeline_object, experiment_name=experiment_name
    # )
    # # wait for the pipeline job to complete
    # workspace_ml_client.jobs.stream(huggingface_pipeline_job.name)

    # download_path = "./pipeline_outputs/"

    # # delete the folder if already exists
    # if os.path.exists(download_path):
    #     shutil.rmtree(download_path)

    # # if pipeline job was not scheduled, skip
    # if huggingface_pipeline_job  is not None:

    #     print("Pipeline job: " + huggingface_pipeline_job.name)
    #     print("Downloading pipeline job output: model_registration_details")

    #     pipeline_download_path = os.path.join(download_path, huggingface_pipeline_job.name)
    #     os.makedirs(pipeline_download_path, exist_ok=True)

    #     workspace_ml_client.jobs.download(
    #         name=huggingface_pipeline_job.name,
    #         download_path=pipeline_download_path,
    #         output_name="model_registration_details",
    #     )


    # # if pipeline job was not scheduled, skip
    # if huggingface_pipeline_job is not None:

    #     with open(
    #         f"./pipeline_outputs/{huggingface_pipeline_job.name}/named-outputs/model_registration_details/model_registration_details.json",
    #         "r",
    #     ) as f:
    #         registration_details = json.load(f)

    #     model_name = registration_details["name"]
    #     model_version = registration_details["version"]

        # Get the model object from workspace
        # model = workspace_ml_client.models.get(name=model_name, version=model_version)
        model = workspace_ml_client.models.get(test_model_name, label="latest")
        # print(f"\n{model_name}")
        print(model.__dict__)
        HF=model.flavors is None
        if HF==False:
            ML=model.flavors['python_function']['loader_module']
            try:
                ML='mlflow.transformers'
                print("ML Model")
            except:
                raise Exception('Some message')
            else:
                raise Exception('Some message')
