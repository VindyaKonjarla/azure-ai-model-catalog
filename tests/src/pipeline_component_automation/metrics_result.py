import pandas as pd
from utils.logging import get_logger


logger = get_logger(__name__)

class MetricsCalaulator:
    def __init__(self, pipeline_jobs, mlflow, experiment_name) -> None:
        self.pipeline_jobs = pipeline_jobs
        self.mlflow = mlflow
        self.experiment_name = experiment_name

    def display_metric(self) -> pd.DataFrame:
        metrics_df = pd.DataFrame()
        for job in self.pipeline_jobs:
            logger.info(job)
            logger.info(job['model_name'][24:])
            # concat 'tags.mlflow.rootRunId=' and pipeline_job.name in single quotes as filter variable
            filter = "tags.mlflow.rootRunId='" + job["job_name"] + "'"
            runs = self.mlflow.search_runs(
                experiment_names=[
                    self.experiment_name], filter_string=filter, output_format="list"
            )
            # get the compute_metrics runs.
            # using a hacky way till 'Bug 2320997: not able to show eval metrics in FT notebooks - mlflow client now showing display names' is fixed
            for run in runs:
                if len(run.data.metrics) > 0:
                    logger.info(run.data.metrics)
                logger.info(run.data.metrics)
                # else, check if run.data.metrics.accuracy exists
                if "exact_match" in run.data.metrics:
                    # get the metrics from the mlflow run
                    run_metric = run.data.metrics
                    # add the model name to the run_metric dictionary
                    run_metric["model_name"] = job["model_name"]
                    # convert the run_metric dictionary to a pandas dataframe
                    temp_df = pd.DataFrame(run_metric, index=[0])
                    # concat the temp_df to the metrics_df
                    metrics_df = pd.concat(
                        [metrics_df, temp_df], ignore_index=True)
        return metrics_df
