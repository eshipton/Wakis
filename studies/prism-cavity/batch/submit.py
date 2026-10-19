"""submit.py RUN_ID STUDY SCRIPT MACHINE "TASKS..." : ship code + submit a Cloud Batch job."""
import json, os, subprocess, sys, tempfile

run_id, study, script, machine, tasks = sys.argv[1:6]
bucket = "cloud-code-511302-autoresearch"
root = os.path.expanduser("~/computational-electromagnetics/wt-prism")
here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
files = subprocess.run("git ls-files; git ls-files --others --exclude-standard", shell=True,
                       cwd=root, capture_output=True, text=True, check=True).stdout.split()
files = sorted({f for f in files if os.path.isfile(os.path.join(root, f)) and "/data/" not in f})
lst = os.path.join(tmp, "files.txt")
open(lst, "w").write("\n".join(files))
tgz = os.path.join(tmp, "code.tgz")
subprocess.run(["tar", "--no-xattrs", "-czf", tgz, "-T", lst], cwd=root, check=True,
               env=dict(os.environ, COPYFILE_DISABLE="1"))
base = f"gs://{bucket}/sweeps/{run_id}"
subprocess.run(["gcloud", "storage", "cp", tgz, f"{base}/code.tgz", "--quiet"], check=True)
subprocess.run(["gcloud", "storage", "cp", os.path.join(here, "task.sh"), f"{base}/task.sh", "--quiet"], check=True)
n = len(tasks.split())
mem = {"n2-standard-2": 7000, "n2-standard-4": 15000, "n2-highmem-2": 15000, "n2-highmem-4": 31000, "n2-standard-8": 30000}[machine]
cpu = int(machine.split("-")[-1]) * 1000
job = {
    "taskGroups": [{
        "taskCount": n, "parallelism": n,
        "taskSpec": {
            "runnables": [{"container": {
                "imageUri": "us-central1-docker.pkg.dev/cloud-code-511302/autoresearch/probe:latest",
                "entrypoint": "/bin/bash",
                "commands": ["-c", f"gcloud storage cp {base}/task.sh /work/task.sh --quiet && bash /work/task.sh"]}}],
            "computeResource": {"cpuMilli": cpu, "memoryMib": mem},
            "maxRunDuration": "25200s", "maxRetryCount": 2,
            "lifecyclePolicies": [{"action": "RETRY_TASK", "actionCondition": {"exitCodes": [50001]}}],
            "environment": {"variables": {"RUN_ID": run_id, "BUCKET": bucket, "STUDY": study,
                                          "SCRIPT": script, "TASKS": tasks}}}}],
    "allocationPolicy": {"instances": [{"policy": {"machineType": machine, "provisioningModel": "SPOT",
                                                   "bootDisk": {"sizeGb": 40}}}],
                         "serviceAccount": {"email": "autoresearch-runner@cloud-code-511302.iam.gserviceaccount.com"}},
    "logsPolicy": {"destination": "CLOUD_LOGGING"},
    "labels": {"autoresearch": "sweep", "tag": "prism"},
}
jf = os.path.join(tmp, "job.json")
json.dump(job, open(jf, "w"), indent=1)
subprocess.run(["gcloud", "batch", "jobs", "submit", run_id, "--location", "us-central1",
                "--project", "cloud-code-511302", "--config", jf], check=True)
print("submitted", run_id, n, "tasks on", machine)
