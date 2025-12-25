from os import system
from pathlib import Path
from subprocess import PIPE, Popen
from time import sleep

HOSTS_CNT = 17
V100S_32G = [1, 2, 3, 4, 5, 6, 11, 12]
V100_32G = [7, 8]
V100_16G = [9, 10]
A100_40G = [13]
L20_48G = [14, 15, 16, 17]

EXP_DIR = "conf"
ORDER = "r15s"
INCLUDE = None
EXCLUDE = None
INTERVAL = 15

GPU_MODE = "exclusive_process"  # shared, exclusive_process


def is_job_pending():
    res = Popen("bjobs -p", shell=True, stdout=PIPE, stderr=PIPE).communicate()[1]
    return b"No pending job found" not in res


def get_hosts() -> tuple[bool, set[int]]:
    in_hosts = set(INCLUDE) if INCLUDE else range(1, HOSTS_CNT + 1)
    ex_hosts = set(EXCLUDE) if EXCLUDE else set()

    in_hosts = set(in_hosts) - ex_hosts
    ex_hosts = set(range(1, HOSTS_CNT + 1)) - in_hosts

    if len(in_hosts) > len(ex_hosts):
        return False, ex_hosts
    else:
        return True, in_hosts


dir = Path(EXP_DIR)
files = [str(f) for f in dir.glob("*.json")] + [str(f) for f in dir.glob("*.toml")]


for i, file in enumerate(files, start=1):
    print(f"Processing {file} ({i}/{len(files)})\n")

    if "local" in file:
        print(f"Skipping local job {file}\n")
        continue

    wait_cnt = 0
    while is_job_pending():
        wait_cnt += 1
        print(f"There are pending jobs, waiting... ({wait_cnt * INTERVAL}s elapsed)\n")
        sleep(INTERVAL)

    name = Path(file).stem

    use_in_hosts, hosts = get_hosts()
    select_list = [f"hname{'==' if use_in_hosts else '!='}gpu{gid:02}" for gid in hosts]

    args = [
        f"-gpu 'num=1:mode={GPU_MODE}'",
        f"-R 'order[{ORDER}]'",
        f"-R 'select[{(' || ' if use_in_hosts else ' && ').join(select_list)}]'" if select_list else "",
        f"-J {name}",
        f"-oo output/{name}.out",
        f"-eo output/{name}.err",
    ]
    cmd = f"uv run src/main.py {file}"
    full_cmd = f"bsub {' '.join(args)} {cmd}"

    print(f"{full_cmd}\n")
    system(full_cmd)
    print()

    sleep(INTERVAL)

    system("bjobs | sort -k1,1n")
    print()
