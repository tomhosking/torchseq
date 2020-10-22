import os
import logging
import requests


def update_mckenzie(progress, metric):
    logger = logging.getLogger("McKenzie")
    try:
        job_id = os.environ["SLURM_JOB_ID"]
        partition = os.environ["SLURM_JOB_PARTITION"]
        endpoint = os.environ["MCKENZIE_ENDPOINT"]

        requests.post(
            "http://" + endpoint + "/hooks/update_job/",
            data={"jobid": job_id, "partition": partition, "progress": progress, "metric": metric},
        )
    except Exception as e:
        logger.error("Error updating McKenzie: " + repr(e))
