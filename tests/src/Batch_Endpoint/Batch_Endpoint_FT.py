from transformers import AutoTokenizer
from azureml.core import Workspace, Environment
#from batch_inference_and_deployment import BatchDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import Input, MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
from azureml.core.compute import AmlCompute
from huggingface_hub import login
from azureml.core.compute_target import ComputeTargetException
from azure.ai.ml.constants import AssetTypes
from mlflow.tracking.client import MlflowClient
import time
from fetch_tasks import HfTask
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    Model,
)
from azureml.core.datastore import Datastore
from azureml.core import Workspace
from mlflow.tracking.client import MlflowClient
import re
from datetime import datetime
