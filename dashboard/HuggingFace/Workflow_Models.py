import os,sys
import requests
import pandas
from datetime import datetime
from github import Github, Auth

 

class Dashboard():
    def __init__(self): 
        self.github_token = os.environ['token']
        #self.github_token = "API_TOKEN"
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)
        self.repo = self.auth.get_repo("Azure/azure-ai-model-catalog")
        self.repo_full_name = self.repo.full_name
        self.data = {
            "workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [],
            "updated_at": [], "status": [], "conclusion": [], "jobs_url": []
        }
        self.models_data = []  # Initialize models_data as an empty list

    def get_all_workflow_names(self):
        # workflow_name = ["MLFlow-mosaicml/mpt-30b-instruct"]
        API = "https://api.github.com/repos/Azure/azure-ai-model-catalog/actions/workflows"
        print (f"Getting github workflows from {API}")
        total_pages = None
        current_page = 1
        per_page = 100
        workflow_name = []
        while total_pages is None or current_page <= total_pages:

            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            params = { "per_page": per_page, "page": current_page }
            response = requests.get(API, headers=headers, params=params)
            if response.status_code == 200:
                workflows = response.json()
                # append workflow_runs to runs list
                for workflow in workflows["workflows"]:
                    if (workflow["name"].startswith("oss-base") | workflow["name"].startswith("oss-train") | workflow["name"].startswith("hf-base") | workflow["name"].startswith("hf-train")):
                        workflow_name.append(workflow["name"])
                if not workflows["workflows"]:
                    break
                # workflow_name.extend(json_response['workflows["name"]'])
                if current_page == 1:
                # divide total_count by per_page and round up to get total_pages
                    total_pages = int(workflows['total_count'] / per_page) + 1
                current_page += 1
                # print a single dot to show progress
                print (f"\rWorkflows fetched: {len(workflow_name)}", end="", flush=True)
            else:
                print (f"Error: {response.status_code} {response.text}")
                exit(1)
        print (f"\n")
        # create ../logs/get_github_workflows/ if it does not exist
        # if not os.path.exists("../logs/get_all_workflow_names"):
        #     os.makedirs("../logs/get_all_workflow_names")
        # # dump runs as json file in ../logs/get_github_workflows folder with filename as DDMMMYYYY-HHMMSS.json
        # with open(f"../logs/get_all_workflow_names/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        #     json.dump(workflow_name, f, indent=4)
        return workflow_name



    def workflow_last_run(self): 
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/","-") for workflow_name in workflows_to_include]
        # normalized_workflows = [hf_name for hf_name in workflows_to_include]
        # hf_name = [hf_name for hf_name in workflows_to_include]
        #print(workflow_name)
        # print(hf_name)
        for workflow_name in normalized_workflows:
            try:
                workflow_runs_url = f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/runs"
                response = requests.get(workflow_runs_url, headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                response.raise_for_status()
                runs_data = response.json()

 

                # if "workflow_runs" not in runs_data:
                #     print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                #     continue

 

                workflow_runs = runs_data["workflow_runs"]
                if not workflow_runs:
                    print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                    continue

 

                last_run = workflow_runs[0]
                jobs_response = requests.get(last_run["jobs_url"], headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                jobs_data = jobs_response.json()

 

               # badge_url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/badge.svg"
                html_url = jobs_data["jobs"][0]["html_url"] if jobs_data.get("jobs") else ""

 
                #self.data["workflow_name_mp"] = data["workflow_name"].startswith("MLFlow-MP") == True 
                #self.data["workflow_name_di"] = data["workflow_name"].startswith("MLFlow-DI") == True 
                self.data["workflow_id"].append(last_run["workflow_id"])
                self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                #self.data["workflow_name_di"].append(workflow_name.replace(".yml", ""))
                self.data["last_runid"].append(last_run["id"])
                self.data["created_at"].append(last_run["created_at"])
                self.data["updated_at"].append(last_run["updated_at"])
                self.data["status"].append(last_run["status"])
                self.data["conclusion"].append(last_run["conclusion"])
                self.data["jobs_url"].append(html_url)

 

                #if html_url:
                    #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({html_url})")
                #else:
                    #url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml"
                    #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({url})")
                run_link = f"https://github.com/{self.repo_full_name}/actions/runs/{last_run['id']}"
                models_entry = {
                    "Model": workflow_name.replace(".yml", ""),
                    # "HFLink": f"[Link](https://huggingface.co/{workflow_name.replace(".yml", "").replace("MLFlow-","")})",
                    # "Status": "<span style='background-color: #00FF00; padding: 2px 6px; border-radius: 3px;'>PASS</span>" if last_run["conclusion"] == "success" else "<span style='background-color: #FF0000; padding: 2px 6px; border-radius: 3px;'>FAIL</span>",
                    # "Status": " ✅ PASS" if last_run["conclusion"] == "success" elif last_run["conclusion"] == "failure" "❌ FAIL",
                    "Status": f"{'✅ PASS' if last_run['conclusion'] == 'success' else '❌ FAIL' if last_run['conclusion'] == 'failure' else '🚫 CANCELLED' if last_run['conclusion'] == 'cancelled' else '⏳ RUNNING'}",
                    "LastRunLink": f"[Link]({run_link})",
                    "LastRunTimestamp": last_run["created_at"],
                    "Category": f"""{'Oss-Base' if workflow_name.startswith("oss-base") == True else 'Oss-Train' if workflow_name.startswith("oss-train") == True else 'Hf-Base' if workflow_name.startswith("hf-base") == True else 'Hf-Train' if workflow_name.startswith("hf-train") == True  else 'None' }"""
                }

                self.models_data.append(models_entry)

 

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

 
        # self.models_data.sort(key=lambda x: x["Status"])
        self.models_data.sort(key=lambda x: (x["Status"] != "❌ FAIL", x["Status"]))
        return self.data

    def results(self, last_runs_dict):
        results_dict = {"total_oss-base": 0, "success_oss-base": 0, "failure_oss-base": 0, "cancelled_oss-base": 0,"running_oss-base":0, "not_tested_oss-base": 0, "total_duration_oss-base": 0,
                       "total_oss-train": 0, "success_oss-train": 0, "failure_oss-train": 0, "cancelled_oss-train": 0,"running_oss-train":0, "not_tested_oss-train": 0, "total_duration_oss-train": 0,
                       "total_hf-base": 0, "success_hf-base": 0, "failure_hf-base": 0, "cancelled_hf-base": 0,"running_hf-base":0, "not_tested_hf-base": 0, "total_duration_hf-base": 0,
                       "total_hf-train": 0, "success_hf-train": 0, "failure_hf-train": 0, "cancelled_hf-train": 0,"running_hf-train":0, "not_tested_hf-train": 0, "total_duration_hf-train": 0,
                       }
        summary = []

 

        df = pandas.DataFrame.from_dict(last_runs_dict)
        # df = df.sort_values(by=['status'], ascending=['failure' in df['status'].values])
      
        results_dict["total_oss-base"] =  df.loc[df["workflow_name"].str.startswith("oss-base") == True]["workflow_id"].count()
        results_dict["success_oss-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("oss-base") == True)]['workflow_id'].count()
        results_dict["failure_oss-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("oss-base") == True)]['workflow_id'].count()
        results_dict["cancelled_oss-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("oss-base") == True)]['workflow_id'].count()
        results_dict["running_oss-base"] = df.loc[(df['status'] == 'in_progress') & (df["workflow_name"].str.startswith("oss-base") == True)]['workflow_id'].count()  # Add running count
        # results_dict["not_tested_mp"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()
        results_dict["not_tested_oss-base"] = results_dict["total_oss-base"] - (results_dict["success_oss-base"] + results_dict["failure_oss-base"] + results_dict["cancelled_oss-base"] + results_dict["running_oss-base"])

         
        results_dict["total_oss-train"] = df.loc[df["workflow_name"].str.startswith("oss-train") == True]["workflow_id"].count()
        results_dict["success_oss-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("oss-train") == True)]['workflow_id'].count()
        results_dict["failure_oss-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("oss-train") == True)]['workflow_id'].count()
        results_dict["cancelled_oss-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("oss-train") == True)]['workflow_id'].count()
        results_dict["running_oss-train"] = df.loc[(df['status'] == 'in_progress')& (df["workflow_name"].str.startswith("oss-train") == True)]['workflow_id'].count()  # Add running count
        # results_dict["not_tested_di"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("oss-train") == True)]['workflow_id'].count()
        results_dict["not_tested_oss-train"] = results_dict["total_oss-train"] - (results_dict["success_oss-train"] + results_dict["failure_oss-train"] + results_dict["cancelled_oss-train"] + results_dict["running_oss-train"])

     
        results_dict["total_hf-base"] = df.loc[df["workflow_name"].str.startswith("hf-base") == True]["workflow_id"].count()
        results_dict["success_hf-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("hf-base") == True)]['workflow_id'].count()
        results_dict["failure_hf-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("hf-base") == True)]['workflow_id'].count()
        results_dict["cancelled_hf-base"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("hf-base") == True)]['workflow_id'].count()
        results_dict["running_hf-base"] = df.loc[(df['status'] == 'in_progress')& (df["workflow_name"].str.startswith("hf-base") == True)]['workflow_id'].count()  # Add running count
        # results_dict["not_tested_import"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("MLFlow-Import") == True)]['workflow_id'].count()
        results_dict["not_tested_hf-base"] = results_dict["total_hf-base"] - (results_dict["success_hf-base"] + results_dict["failure_hf-base"] + results_dict["cancelled_hf-base"] + results_dict["running_import"])

        
        results_dict["total_hf-train"] = df.loc[df["workflow_name"].str.startswith("hf-train") == True]["workflow_id"].count()
        results_dict["success_hf-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("hf-train") == True)]['workflow_id'].count()
        results_dict["failure_hf-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("hf-train") == True)]['workflow_id'].count()
        results_dict["cancelled_hf-train"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("hf-train") == True)]['workflow_id'].count()
        results_dict["running_hf-train"] = df.loc[(df['status'] == 'in_progress')& (df["workflow_name"].str.startswith("hf-train") == True)]['workflow_id'].count()  # Add running count
        # results_dict["not_tested_batch"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("MLFlow-Batch") == True)]['workflow_id'].count()
        results_dict["not_tested_hf-train"] = results_dict["total_hf-train"] - (results_dict["success_hf-train"] + results_dict["failure_hf-train"] + results_dict["cancelled_hf-train"] + results_dict["running_hf-train"])

     
  

     
        summary.append("|Category|🚀Total|✅Pass|Pass%|❌Failure|Failure%|🚫Cancelled|⏳Running|Others") 
        summary.append("| ----------- | ----------------- | -------- | -------- | --------  | -------- | --------- | ---------- | -----------|")
        #summary.append("| Online Endpoint Deployment - Dynamic Installation| ")      
        #summary.append("| Online Endpoint Deployment - Packaging| )
        #summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        #summary.append("-----|-------|-------|-------|-------|")
        
        summary.append(f"Oss-Base|{results_dict['total_oss-base']}|{results_dict['success_oss-base']}|{results_dict['success_oss-base']/results_dict['total_oss-base']:.2%}|{results_dict['failure_oss-base']}|{results_dict['failure_oss-base']/results_dict['total_oss-base']:.2%}|{results_dict['cancelled_oss-base']}|{results_dict['running_oss-base']}|{results_dict['not_tested_oss-base']}|")
        summary.append(f"Oss-Train|{results_dict['total_oss-train']}|{results_dict['success_oss-train']}|{results_dict['success_oss-train']/results_dict['total_oss-train']:.2%}|{results_dict['failure_oss-train']}|{results_dict['failure_oss-train']/results_dict['total_oss-train']:.2%}|{results_dict['cancelled_oss-train']}|{results_dict['running_oss-train']}|{results_dict['not_tested_oss-train']}|")
        summary.append(f"Hf-Base|{results_dict['total_hf-base']}|{results_dict['success_hf-base']}|{results_dict['success_hf-base']/results_dict['total_hf-base']:.2%}|{results_dict['failure_hf-base']}|{results_dict['failure_hf-base']/results_dict['total_hf-base']:.2%}|{results_dict['cancelled_hf-base']}|{results_dict['running_hf-base']}|{results_dict['not_tested_hf-base']}|")
        summary.append(f"Hf-Train|{results_dict['total_hf-train']}|{results_dict['success_hf-train']}|{results_dict['success_hf-train']/results_dict['total_hf-train']:.2%}|{results_dict['failure_hf-train']}|{results_dict['failure_hf-train']/results_dict['total_hf-train']:.2%}|{results_dict['cancelled_hf-train']}|{results_dict['running_hf-train']}|{results_dict['not_tested_hf-train']}|")
        #summary.append(f"Evaluate|{results_dict['total_eval']}|{results_dict['success_eval']}|{results_dict['success_eval']/results_dict['total_eval']:.2%}|{results_dict['failure_eval']}|{results_dict['failure_eval']/results_dict['total_eval']:.2%}|{results_dict['cancelled_eval']}|{results_dict['running_eval']}|{results_dict['not_tested_eval']}|")
     
        models_df = pandas.DataFrame.from_dict(self.models_data)
        models_md = models_df.to_markdown()

 

        summary_text = "\n".join(summary)
       
        with open("Workflow_Models.md", "w", encoding="utf-8") as f:
            f.write(summary_text)
            f.write(os.linesep)
            f.write(os.linesep)
            f.write(models_md)

 

def main():

        my_class = Dashboard()
        last_runs_dict = my_class.workflow_last_run()
        my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
