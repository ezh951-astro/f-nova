import os
import subprocess
import time
import datetime

def insert_yaml_block( insertlines, targetfile, location ):

    with open(targetfile, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if location in line: # will insert before location
            lines[i:i] = insertlines
            break

    with open(targetfile, "w") as f:
        f.writelines(lines)

def check_if_recently_generated(path, max_age_seconds=30):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
    
    if age.total_seconds() > max_age_seconds:
        raise RuntimeError(f"file is {age.total_seconds():.1f}s old, expected fresh file")

def poll_slurm_queue(run_name):

    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--format=%.10i %.30j %.10P %.8u %.2t %.12M %.5D"],
        capture_output=True,
        text=True
    )

    lines = result.stdout.splitlines()
    if len(lines) <= 1:
        return []
    
    header = lines[0].split()

    jobs = [
        dict(zip(header, line.split()))
        for line in lines[1:]
        if line.strip()
    ]

    return next( (job for job in jobs if job["NAME"]==run_name), None )

def ensure_queue_submission(run_name, timeout=60, poll_interval=1.0):

    jobstate = None

    deadline = time.time() + timeout  # seconds
    while jobstate not in ["PD", "R"]:

        if time.time() > deadline:
            raise TimeoutError(f"Job '{run_name}' did not appear in queue within {timeout}s")

        print(f"Waiting for job {run_name} to hit the queue...")

        job = poll_slurm_queue(run_name)
        if job is not None:
            jobstate = job["ST"]

        time.sleep(poll_interval)