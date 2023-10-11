from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.ai.ml.dsl import pipeline
from huggingface_hub import HfApi
import re
import ast
import os
import pandas as pd
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
from azure.ai.ml.entities import AmlCompute
import time
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
	    subscription_id = "80c77c76-74ba-4c8c-8229-4c3b2957990c",
        resource_group_name = "huggingface-registry-test1",
        workspace_name = "test-koreacentral"
	)
ml_client_registry = MLClient(credential, registry_name="azureml-preview-test1")
model_name = "databricks-dolly-v2-12b"
foundation_model = workspace_ml_client.models.get(model_name, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
try: foundation_model.flavors['python_function']['loader_module']=='mlflow.transformers'
    print("Model is in mlflow")
except ResourceNotFoundError: 
    raise Exception('Some message')
