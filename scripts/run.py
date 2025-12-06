from os import system
from pathlib import Path
from subprocess import PIPE, Popen
from time import sleep

EXP_DIR = "conf"
ORDER = "r15s"
HOSTS_CNT = 17
INCLUDE = None
# EXCLUDE = None
EXCLUDE = [7, 8, 9, 10, 13, 14, 15, 16, 17]
INTERVAL = 15


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
files = [str(f) for f in dir.glob("*.json")]

for i, file in enumerate(files, start=1):
    print(f"Processing {file} ({i}/{len(files)})\n")
    while is_job_pending():
        sleep(10)

    name = Path(file).stem

    use_in_hosts, hosts = get_hosts()
    select_list = [f"hname{'==' if use_in_hosts else '!='}gpu{gid:02}" for gid in hosts]

    args = [
        "-gpu 'num=1:mode=exclusive_process'",
        f"-R 'order[{ORDER}]'",
        f"-R 'select[{' && '.join(select_list)}]'",
        f"-J {name}",
        f"-oo output/{name}.out",
        f"-eo output/{name}.err",
    ]
    cmd = f"python src/main.py {file}"
    full_cmd = f"bsub {' '.join(args)} {cmd}"

    print(f"{full_cmd}\n")
    system(full_cmd)
    print()

    sleep(INTERVAL)

    system("bjobs | sort -k1,1n")
    print()
