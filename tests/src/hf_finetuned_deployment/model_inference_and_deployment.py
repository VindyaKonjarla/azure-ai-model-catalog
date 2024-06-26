import time
import json
import os
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings
)
from utils.logging import get_logger
from fetch_task import HfTask
import mlflow
from box import ConfigBox
import re
import sys
from transformers import AutoTokenizer

logger = get_logger(__name__)


class ModelInferenceAndDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client

    def get_error_messages(self):
        # load ../../config/errors.json into a dictionary
        with open('../../config/errors.json') as f:
            return json.load(f)

    def prase_logs(self, logs):
        error_messages = self.get_error_messages()
        # split logs by \n
        logs_list = logs.split("\n")
        # loop through each line in logs_list
        for line in logs_list:
            # loop through each error in errors
            for error in error_messages:
                # if error is found in line, print error message
                if error['parse_string'] in line:
                    logger.error(
                        f"::error:: {error_messages['error_category']}: {line}")

    def get_online_endpoint_logs(self, online_endpoint_name, deployment_name):
        logger.info("Deployment logs: \n\n")
        logs = self.workspace_ml_client.online_deployments.get_logs(
            name=deployment_name, endpoint_name=online_endpoint_name, lines=100000)
        print(logs)
        self.prase_logs(logs)

    def get_task_specified_input(self, task, actual_model_name):
        #scoring_file = f"../../config/sample_inputs/{self.registry}/{task}.json"
        scoring_file = f"sample_inputs/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
            if task == "fill-mask":
                tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
                for index in range(len(scoring_input.input_data)):
                    scoring_input.input_data[index] = scoring_input.input_data[index].replace(
                        "<mask>", tokenizer.mask_token).replace("[MASK]", tokenizer.mask_token)
                scoring_file, scoring_input = self.create_json_file(file_name=f"{task}-alternate.json", dicitonary=scoring_input)
            logger.info(f"scoring_input file:\n\n {scoring_input}\n\n")  
        except Exception as e:
            logger.error(
                f"::Error:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")
        return scoring_file, scoring_input

    def create_json_file(self, file_name, dicitonary):
        logger.info("Inside the create json file method...")
        try:
            json_file_name = file_name+".json"
            save_file = open(json_file_name, "w")
            json.dump(dicitonary, save_file, indent=4)
            save_file.close()
            logger.info(
                f"Successfully creating the json file with name {json_file_name}")
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Getting error while creating and saving the jsonfile, the error is occuring at this line no : {exc_tb.tb_lineno}" +
                f"reason is this : \n {ex}")
            raise Exception(ex)
        json_obj = json.dumps(dicitonary, indent=4)
        scoring_input = ConfigBox(json.loads(json_obj))
        logger.info(f"Our new scoring input is this one : {scoring_input}")
        return json_file_name, scoring_input

    def delete_file(self, file_name):
        logger.info("Started deleting the file...")
        os.remove(path=file_name)
    
    def get_task_params(self) -> ConfigBox:
        queue_file = f"task_params.json"
        with open(queue_file) as f:
            return ConfigBox(json.load(f))

    def cloud_inference(self, scoring_file, scoring_input, online_endpoint_name, deployment_name):
        try:
            json_file_name = ''
            logger.info(f"endpoint_name : {online_endpoint_name}")
            logger.info(f"deployment_name : {deployment_name}")
            logger.info(f"Input data is this one : {scoring_input}")
            try:
                configbox_obj = self.get_task_params()
                input_data = configbox_obj.get(self.test_model_name, None)
                if input_data == None:
                    response = self.workspace_ml_client.online_endpoints.invoke(
                        endpoint_name=online_endpoint_name,
                        deployment_name=deployment_name,
                        request_file=scoring_file,
                    )
                else:
                    json_file_name, scoring_input = self.create_json_file(
                        file_name=deployment_name, dicitonary=input_data)
                    logger.info("Online endpoint invoking satrted...")
                    response = self.workspace_ml_client.online_endpoints.invoke(
                        endpoint_name=online_endpoint_name,
                        deployment_name=deployment_name,
                        request_file=json_file_name,
                    )
                logger.info(
                    f"Getting the reposne from the endpoint is this one : {response}")
            except Exception as ex:
                logger.warning(
                    "::warning:: Trying to invoking the endpoint again by changing the input data and file")
                logger.warning(
                    f"::warning:: This is failed due to this :\n {ex}")
                sys.exit(1)

            response_json = json.loads(response)
            output = json.dumps(response_json, indent=2)
            logger.info(f"response: \n\n{output}")
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fh:
                print(f'####Sample input', file=fh)
                print(f'```json', file=fh)
                print(f'{scoring_input}', file=fh)
                print(f'```', file=fh)
                print(f'####Sample output', file=fh)
                print(f'```json', file=fh)
                print(f'{output}', file=fh)
                print(f'```', file=fh)
        except Exception as e:
            if os.path.exists(json_file_name):
                logger.info(f"Deleting the json file : {json_file_name}")
                self.delete_file(file_name=json_file_name)
            logger.error(f"::error:: Could not invoke endpoint: \n")
            logger.info(f"::error::The exception here is this : \n {e}")
            raise Exception(e)

    def create_online_endpoint(self, endpoint):
        logger.info("In create_online_endpoint...")
        try:
            self.workspace_ml_client.online_endpoints.begin_create_or_update(
                endpoint).wait()
        except Exception as e:
            logger.error(f"::error:: Could not create endpoint: \n")
            logger.error(f"{e}\n\n check logs:\n\n")
            self.prase_logs(str(e))
            exit(1)
        online_endpoint_obj = self.workspace_ml_client.online_endpoints.get(
            name=endpoint.name)
        logger.info(f"online_endpoint_obj : {online_endpoint_obj}")

    def get_model_name(self, latest_model_name):
        # Expression need to be replaced with hyphen
        expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                                "$", "%", "^", "&", "*", "<", ">", "?", "!", "~", "_"]
        # Create the regular expression to ignore
        regx_for_expression = re.compile(
            '|'.join(map(re.escape, expression_to_ignore)))
        # Check the model_name contains any of there character
        expression_check = re.findall(regx_for_expression, latest_model_name)
        if expression_check:
            # Replace the expression with hyphen
            latest_model_name = regx_for_expression.sub("-", latest_model_name)
        # Reserve Keyword need to be removed
        reserve_keywords = ["microsoft"]
        # Create the regular expression to ignore
        regx_for_reserve_keyword = re.compile(
            '|'.join(map(re.escape, reserve_keywords)))
        # Check the model_name contains any of the string
        reserve_keywords_check = re.findall(
            regx_for_reserve_keyword, latest_model_name)
        if reserve_keywords_check:
            # Replace the resenve keyword with nothing with hyphen
            latest_model_name = regx_for_reserve_keyword.sub(
                '', latest_model_name)
            latest_model_name = latest_model_name.lstrip("-")

        return latest_model_name.lower()
    
    def create_online_deployment(self, latest_model, online_endpoint_name, instance_type, endpoint):
        logger.info("In create_online_deployment...")
        logger.info(f"latest_model.name is this : {latest_model.name}")
        latest_model_name = self.get_model_name(
            latest_model_name=latest_model.name)
        # Check if the model name starts with a digit
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
        # Check the model name is more then 32 character
        if len(latest_model.name) > 28:
            model_name = latest_model_name[:28]
            deployment_name = model_name.rstrip("-")
        else:
            deployment_name = latest_model_name
        deployment_name = deployment_name + "-di"
        logger.info(f"deployment name is this one : {deployment_name}")
        deployment_config = ManagedOnlineDeployment(
            name=deployment_name,
            model=latest_model.id,
            endpoint_name=online_endpoint_name,
            instance_type=instance_type,
            instance_count=1,
            request_settings=OnlineRequestSettings(
                max_concurrent_requests_per_instance=1,
                request_timeout_ms=90000,
                max_queue_wait_ms=500,
            ),
            liveness_probe=ProbeSettings(
            failure_threshold=30,
            success_threshold=1,
            timeout=2,
            period=10,
            initial_delay=2000,
            ),
            readiness_probe=ProbeSettings(
            failure_threshold=10,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=2000,
            )
        )
        try:
            self.workspace_ml_client.online_deployments.begin_create_or_update(
                deployment_config).wait()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create deployment\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            self.prase_logs(str(e))
            self.get_online_endpoint_logs(
                online_endpoint_name, deployment_name)
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
            sys.exit(1)
        endpoint.traffic = {deployment_name: 100}
        try:
            self.workspace_ml_client.begin_create_or_update(endpoint).result()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create deployment\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            self.get_online_endpoint_logs(
                online_endpoint_name, deployment_name)
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=endpoint.name).wait()
            sys.exit(1)
        deployment_obj = self.workspace_ml_client.online_deployments.get(
            name=deployment_name, endpoint_name=endpoint.name)
        logger.info(f"Deployment object is this one: {deployment_obj}")
        return deployment_name

    # def delete_online_endpoint(self, online_endpoint_name):
    #     try:
    #         logger.info("\n In delete_online_endpoint.....")
    #         self.workspace_ml_client.online_endpoints.begin_delete(
    #             name=online_endpoint_name).wait()
    #     except Exception as e:
    #         _, _, exc_tb = sys.exc_info()
    #         logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
    #                      f" the exception is this one : {e}")
    #         logger.error(f"::warning:: Could not delete endpoint: : \n{e}")
    #         exit(0)
            
    def delete_online_deployment(self, endpoint, online_endpoint_name, deployment_name):
        try:
            logger.info(
                "Bringing down the live trafic allocation to zero and then update the endpoint")
            endpoint.traffic = {deployment_name: 0}
            self.workspace_ml_client.begin_create_or_update(endpoint).result()
            logger.info("\n Started deleting online_deployment.....")
            self.workspace_ml_client.online_deployments.begin_delete(
                name=deployment_name, endpoint_name=online_endpoint_name).wait()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                "::Error:: Could not delete deployment from the endpoint due to below reason")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            exit(0)

    def model_infernce_and_deployment(self, instance_type, task, latest_model, endpoint, actual_model_name):  
        logger.info(f"latest_model: {latest_model}")
        logger.info(f"Task is : {task}")
        # scoring_file, scoring_input = self.get_task_specified_input(task=task, actual_model_name=actual_model_name)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        # timestamp = int(time.time())
        # online_endpoint_name = task + str(timestamp)
        online_endpoint_name = endpoint.name
        #online_endpoint_name = "Testing" + str(timestamp)
        logger.info(f"online_endpoint_name: {online_endpoint_name}")
        # endpoint = ManagedOnlineEndpoint(
        #     name=online_endpoint_name,
        #     auth_mode="key",
        # )
        logger.info(f"ManageOnlieEndpoint : {endpoint}")
        #self.create_online_endpoint(endpoint=endpoint)
        deployment_name = self.create_online_deployment(
            latest_model=latest_model,
            online_endpoint_name=online_endpoint_name,
            instance_type=instance_type,
            endpoint=endpoint
        )
        # self.cloud_inference(
        #     scoring_file=scoring_file,
        #     scoring_input=scoring_input,
        #     online_endpoint_name=online_endpoint_name,
        #     deployment_name = deployment_name
        # )
        ##self.delete_online_endpoint(online_endpoint_name=online_endpoint_name)
        #self.delete_online_deployment(endpoint=endpoint, online_endpoint_name=online_endpoint_name, deployment_name=deployment_name)
