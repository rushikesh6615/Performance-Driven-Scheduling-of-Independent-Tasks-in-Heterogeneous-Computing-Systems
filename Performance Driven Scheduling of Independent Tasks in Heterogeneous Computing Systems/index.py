import random
import time
import copy
from typing import List
from collections import namedtuple

ScheduleResult = namedtuple("ScheduleResult", ["schedule", "makespan", "execution_time"])

def parse_etcs_1024(filename: str, num_tasks: int, num_procs: int) -> list[list[float]]:
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line
        numbers = [float(line.strip()) for line in lines if line.strip()]
    
    expected_count = num_tasks * num_procs
    if len(numbers) != expected_count:
        raise ValueError(f"Expected {expected_count} numbers, got {len(numbers)}")
    
    return [
        numbers[i * num_procs:(i + 1) * num_procs]
        for i in range(num_tasks)
    ]

def parse_etcs_512(filename: str, num_tasks: int, num_procs: int) -> list[list[float]]:
    with open(filename, 'r') as file:
        lines = file.readlines()[0:]  # Skip the first line
        numbers = [float(line.strip()) for line in lines if line.strip()]
    
    expected_count = num_tasks * num_procs
    if len(numbers) != expected_count:
        raise ValueError(f"Expected {expected_count} numbers, got {len(numbers)}")
    
    return [
        numbers[i * num_procs:(i + 1) * num_procs]
        for i in range(num_tasks)
    ]

def min_min(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    proc_time = [0] * num_procs
    scheduled = [False] * num_tasks
    schedule = [[] for _ in range(num_procs)]

    while not all(scheduled):
        min_ect = float('inf')
        selected_task = selected_proc = -1
        for t in range(num_tasks):
            if not scheduled[t]:
                for p in range(num_procs):
                    ect = proc_time[p] + etc[t][p]
                    if ect < min_ect:
                        min_ect = ect
                        selected_task, selected_proc = t, p
        proc_time[selected_proc] = min_ect
        scheduled[selected_task] = True
        schedule[selected_proc].append(selected_task)

    return ScheduleResult(schedule, max(proc_time), time.time() - start)

def max_min(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    proc_time = [0] * num_procs
    scheduled = [False] * num_tasks
    schedule = [[] for _ in range(num_procs)]

    while not all(scheduled):
        max_min_ect = -1
        selected_task = selected_proc = -1
        for t in range(num_tasks):
            if not scheduled[t]:
                min_ect = float('inf')
                min_p = -1
                for p in range(num_procs):
                    ect = proc_time[p] + etc[t][p]
                    if ect < min_ect:
                        min_ect, min_p = ect, p
                if min_ect > max_min_ect:
                    max_min_ect = min_ect
                    selected_task, selected_proc = t, min_p
        proc_time[selected_proc] += etc[selected_task][selected_proc]
        scheduled[selected_task] = True
        schedule[selected_proc].append(selected_task)

    return ScheduleResult(schedule, max(proc_time), time.time() - start)

def mct(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    schedule = [[] for _ in range(num_procs)]
    proc_time = [0.0] * num_procs

    for t in range(num_tasks):
        min_completion_time = float('inf')
        best_proc = 0
        for p in range(num_procs):
            completion_time = proc_time[p] + etc[t][p]
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                best_proc = p
        schedule[best_proc].append(t)
        proc_time[best_proc] += etc[t][best_proc]

    return ScheduleResult(schedule, max(proc_time), time.time() - start)

def met(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    schedule = [[] for _ in range(num_procs)]
    proc_time = [0.0] * num_procs

    for t in range(num_tasks):
        best_proc = min(range(num_procs), key=lambda p: etc[t][p])
        schedule[best_proc].append(t)
        proc_time[best_proc] += etc[t][best_proc]

    return ScheduleResult(schedule, max(proc_time), time.time() - start)

def min_max(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    assignments = [[] for _ in range(num_procs)]
    proc_times = [0.0] * num_procs
    assigned = [False] * num_tasks

    for _ in range(num_tasks):
        max_ratio = -1.0
        selected_task = selected_proc = -1
        for t in range(num_tasks):
            if assigned[t]:
                continue
            min_completion_time = float('inf')
            best_proc = -1
            for p in range(num_procs):
                completion_time = proc_times[p] + etc[t][p]
                if completion_time < min_completion_time:
                    min_completion_time = completion_time
                    best_proc = p
            min_exec_time = min(etc[t])
            ratio = min_exec_time / max(etc[t][best_proc], 1e-6)  # Safe division
            if ratio > max_ratio:
                max_ratio = ratio
                selected_task = t
                selected_proc = best_proc
        assignments[selected_proc].append(selected_task)
        proc_times[selected_proc] += etc[selected_task][selected_proc]
        assigned[selected_task] = True

    return ScheduleResult(assignments, max(proc_times), time.time() - start)

def sufferage(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    proc_time = [0.0] * num_procs
    scheduled = [False] * num_tasks
    schedule = [[] for _ in range(num_procs)]

    while not all(scheduled):
        max_suffer = -1
        selected_task = selected_proc = -1
        for t in range(num_tasks):
            if scheduled[t]: continue
            ects = [(proc_time[p] + etc[t][p], p) for p in range(num_procs)]
            ects.sort()
            suffer = ects[1][0] - ects[0][0] if len(ects) > 1 else 0
            if suffer > max_suffer:
                max_suffer = suffer
                selected_task, selected_proc = t, ects[0][1]
        schedule[selected_proc].append(selected_task)
        proc_time[selected_proc] += etc[selected_task][selected_proc]
        scheduled[selected_task] = True

    return ScheduleResult(schedule, max(proc_time), time.time() - start)

def list_sufferage(etc: list[list[float]]) -> ScheduleResult:
    start = time.time()
    num_tasks = len(etc)
    num_procs = len(etc[0])

    proc_time = [0.0 for _ in range(num_procs)]
    schedule = [[] for _ in range(num_procs)]
    scheduled = [False for _ in range(num_tasks)]

    sorted_tasks_per_proc = [[] for _ in range(num_procs)]
    min1_proc, min2_proc = [0]*num_tasks, [0]*num_tasks
    min1, min2 = [0.0]*num_tasks, [0.0]*num_tasks

    for t in range(num_tasks):
        sorted_etc = sorted((etc[t][p], p) for p in range(num_procs))
        min1[t], min2[t] = sorted_etc[0][0], sorted_etc[1][0]
        min1_proc[t], min2_proc[t] = sorted_etc[0][1], sorted_etc[1][1]

    for p in range(num_procs):
        for t in range(num_tasks):
            if p == min1_proc[t]:
                priority = min2[t] / max(min1[t], 1e-6)
            else:
                priority = min1[t] / max(etc[t][p], 1e-6)
            sorted_tasks_per_proc[p].append((t, priority))
        sorted_tasks_per_proc[p].sort(key=lambda x: x[1], reverse=True)

    task_indices = [0] * num_procs
    while any(not s for s in scheduled):
        min_eft = float('inf')
        tmin = -1
        pmin = -1

        for p in range(num_procs):
            while task_indices[p] < len(sorted_tasks_per_proc[p]) and scheduled[sorted_tasks_per_proc[p][task_indices[p]][0]]:
                task_indices[p] += 1
            if task_indices[p] >= len(sorted_tasks_per_proc[p]):
                continue

            t = sorted_tasks_per_proc[p][task_indices[p]][0]
            eft = proc_time[p] + etc[t][p]

            if eft < min_eft:
                min_eft = eft
                tmin = t
                pmin = p

        if tmin != -1 and pmin != -1:
            schedule[pmin].append(tmin)
            proc_time[pmin] += etc[tmin][pmin]
            scheduled[tmin] = True

    end = time.time()
    return ScheduleResult(schedule, max(proc_time), end - start)

def pb_algorithm(etc):
    start = time.time()
    num_tasks, num_procs = len(etc), len(etc[0])
    proc_times = [0.0] * num_procs
    assignments = [[] for _ in range(num_procs)]

    for t in range(num_tasks):
        best_proc = min(range(num_procs), key=lambda p: etc[t][p])
        assignments[best_proc].append(t)
        proc_times[best_proc] += etc[t][best_proc]

    for _ in range(num_tasks):
        pa = max(range(num_procs), key=lambda p: proc_times[p])
        pb = min(range(num_procs), key=lambda p: proc_times[p])
        ca, cb = proc_times[pa], proc_times[pb]
        best_task, min_penalty = -1, float('inf')

        for t in assignments[pa]:
            if cb + etc[t][pb] >= (ca + cb) / 2: continue
            penalty = (etc[t][pb] - etc[t][pa]) / max(etc[t][pa], 1e-6)
            if penalty < min_penalty:
                min_penalty, best_task = penalty, t

        if best_task == -1:
            break
        assignments[pa].remove(best_task)
        assignments[pb].append(best_task)
        proc_times[pa] -= etc[best_task][pa]
        proc_times[pb] += etc[best_task][pb]

    return ScheduleResult(assignments, max(proc_times), time.time() - start)

def tenacious_penalty_based(etc: list[list[float]]) -> ScheduleResult:
    start = time.time()
    num_tasks = len(etc)
    num_procs = len(etc[0])

    schedule = [[] for _ in range(num_procs)]
    proc_time = [0.0 for _ in range(num_procs)]

    for t in range(num_tasks):
        best_proc = min(range(num_procs), key=lambda p: etc[t][p])
        schedule[best_proc].append(t)
        proc_time[best_proc] += etc[t][best_proc]

    repetitions = 0
    while repetitions < num_tasks:
        repetitions += 1
        selected_task = -1
        selected_proc = -1
        min_penalty = float('inf')

        sorted_procs = sorted(range(num_procs), key=lambda p: proc_time[p])
        pa = sorted_procs[-1]
        ca = proc_time[pa]

        for pb in sorted_procs:
            if pb == pa:
                continue
            cb = proc_time[pb]

            for t in schedule[pa]:
                etc_pa = etc[t][pa]
                etc_pb = etc[t][pb]
                if cb + etc_pb >= ca:
                    continue
                penalty = (etc_pb - etc_pa) / max(etc_pa, 1e-6)
                if penalty < min_penalty:
                    min_penalty = penalty
                    selected_task = t
                    selected_proc = pb

            if selected_task != -1:
                break

        if selected_task != -1:
            schedule[pa].remove(selected_task)
            proc_time[pa] -= etc[selected_task][pa]
            schedule[selected_proc].append(selected_task)
            proc_time[selected_proc] += etc[selected_task][selected_proc]
        else:
            break

    end = time.time()
    return ScheduleResult(schedule, max(proc_time), end - start)


def tenacious_penalty_based_modified(etc: list[list[float]]) -> ScheduleResult:
    start = time.time()
    num_tasks = len(etc)
    num_procs = len(etc[0])

    schedule = [[] for _ in range(num_procs)]
    proc_time = [0.0 for _ in range(num_procs)]

    for t in range(num_tasks):
        best_proc = min(range(num_procs), key=lambda p: etc[t][p])
        schedule[best_proc].append(t)
        proc_time[best_proc] += etc[t][best_proc]

    repetitions = 0
    while repetitions < num_tasks:
        repetitions += 1
        selected_task = -1
        selected_proc = -1
        min_penalty = float('inf')

        sorted_procs = sorted(range(num_procs), key=lambda p: proc_time[p])
        pa = sorted_procs[-1]
        ca = proc_time[pa]

        for pb in sorted_procs:
            if pb == pa:
                continue
            cb = proc_time[pb]

            for t in schedule[pa]:
                etc_pa = etc[t][pa]
                etc_pb = etc[t][pb]
                if cb + etc_pb >= ca:
                    continue
                penalty = ((etc_pb - etc_pa)*(max(cb + etc_pb,ca-etc_pa)) /ca)/ max(etc_pa, 1e-6)
                if penalty < min_penalty:
                    min_penalty = penalty
                    selected_task = t
                    selected_proc = pb

            if selected_task != -1:
                break

        if selected_task != -1:
            schedule[pa].remove(selected_task)
            proc_time[pa] -= etc[selected_task][pa]
            schedule[selected_proc].append(selected_task)
            proc_time[selected_proc] += etc[selected_task][selected_proc]
        else:
            break

    end = time.time()
    return ScheduleResult(schedule, max(proc_time), end - start)


def tenacious_penalty_based_modified2(etc: list[list[float]]) -> ScheduleResult:
    start = time.time()
    num_tasks = len(etc)
    num_procs = len(etc[0])

    schedule = [[] for _ in range(num_procs)]
    proc_time = [0.0 for _ in range(num_procs)]

    for t in range(num_tasks):
        best_proc = min(range(num_procs), key=lambda p: etc[t][p])
        schedule[best_proc].append(t)
        proc_time[best_proc] += etc[t][best_proc]

    repetitions = 0
    while repetitions < num_tasks:
        repetitions += 1
        selected_task = -1
        selected_proc = -1
        min_penalty = float('inf')

        sorted_procs = sorted(range(num_procs), key=lambda p: proc_time[p])
        pa = sorted_procs[-1]
        ca = proc_time[pa]

        for pb in sorted_procs:
            if pb == pa:
                continue
            cb = proc_time[pb]

            for t in schedule[pa]:
                etc_pa = etc[t][pa]
                etc_pb = etc[t][pb]
                if cb + etc_pb >= ca:
                    continue
                penalty = ((etc_pb - etc_pa)*(cb+etc_pb))/ (max(etc_pa, 1e-6)*ca)
                if penalty < min_penalty:
                    min_penalty = penalty
                    selected_task = t
                    selected_proc = pb

            if selected_task != -1:
                break

        if selected_task != -1:
            schedule[pa].remove(selected_task)
            proc_time[pa] -= etc[selected_task][pa]
            schedule[selected_proc].append(selected_task)
            proc_time[selected_proc] += etc[selected_task][selected_proc]
        else:
            break

    end = time.time()
    return ScheduleResult(schedule, max(proc_time), end - start)

import copy
import copy

def run_all_datasets(datasets, num_tasks, num_procs, parse_func, title=""):
    print(f"\n{title} ({num_tasks}x{num_procs})")
    header = ["Dataset"] + list(algorithms.keys())
    col_width = 20

    # Header
    print(" | ".join(f"{h[:col_width]:<{col_width}}" for h in header))
    print("-" * ((col_width + 3) * len(header)))

    for filename in datasets:
        full_path = f"datasets/{num_tasks}x{num_procs}/{filename}" if "datasets" not in filename else filename
        etc_matrix = parse_func(full_path, num_tasks, num_procs)

        row = [filename]
        for algo in algorithms.values():
            result = algo(copy.deepcopy(etc_matrix))
            row.append(f"{result.makespan:.4f}")
        print(" | ".join(f"{cell:<{col_width}}" for cell in row))

if __name__ == "__main__":
    algorithms = {
        "Min-Min": min_min,
        "Min-Max": min_max,
        "Sufferage": sufferage,
        "List Suff.": list_sufferage,
        "PB Algo": pb_algorithm,
        
        # "TenaciousPBm1.": tenacious_penalty_based_modified,
        "Tenacious PB": tenacious_penalty_based,
        "TenaciousPB Modified.": tenacious_penalty_based_modified2,
    }

    datasets512 = [
        "u_c_hihi.0", "u_c_hilo.0", "u_c_lohi.0", "u_c_lolo.0",
        "u_i_hihi.0", "u_i_hilo.0", "u_i_lohi.0", "u_i_lolo.0",
        "u_s_hihi.0", "u_s_hilo.0", "u_s_lohi.0", "u_s_lolo.0"
    ]

    datasets1024 = [
        "A.u_c_hihi", "A.u_c_hilo", "A.u_c_lohi", "A.u_c_lolo",
        "A.u_i_hihi", "A.u_i_hilo", "A.u_i_lohi", "A.u_i_lolo",
        "A.u_s_hihi", "A.u_s_hilo", "A.u_s_lohi", "A.u_s_lolo",
        "B.u_c_hihi", "B.u_c_hilo", "B.u_c_lohi", "B.u_c_lolo",
        "B.u_i_hihi", "B.u_i_hilo", "B.u_i_lohi", "B.u_i_lolo",
        "B.u_s_hihi", "B.u_s_hilo", "B.u_s_lohi", "B.u_s_lolo"
    ]
    datasets2048 = [
        "A.u_c_hihi", "A.u_c_hilo", "A.u_c_lohi", "A.u_c_lolo",
        "A.u_i_hihi", "A.u_i_hilo", "A.u_i_lohi", "A.u_i_lolo",
        "A.u_s_hihi", "A.u_s_hilo", "A.u_s_lohi", "A.u_s_lolo",
        "B.u_c_hihi", "B.u_c_hilo", "B.u_c_lohi", "B.u_c_lolo",
        "B.u_i_hihi", "B.u_i_hilo", "B.u_i_lohi", "B.u_i_lolo",
        "B.u_s_hihi", "B.u_s_hilo", "B.u_s_lohi", "B.u_s_lolo"
    ]
    datasets4096 = [
        "A.u_c_hihi", "A.u_c_hilo", "A.u_c_lohi", "A.u_c_lolo",
        "A.u_i_hihi", "A.u_i_hilo", "A.u_i_lohi", "A.u_i_lolo",
        "A.u_s_hihi", "A.u_s_hilo", "A.u_s_lohi", "A.u_s_lolo",
        "B.u_c_hihi", "B.u_c_hilo", "B.u_c_lohi", "B.u_c_lolo",
        "B.u_i_hihi", "B.u_i_hilo", "B.u_i_lohi", "B.u_i_lolo",
        "B.u_s_hihi", "B.u_s_hilo", "B.u_s_lohi", "B.u_s_lolo"
    ]
    datasets8192 = [
        "A.u_c_hihi", "A.u_c_hilo", "A.u_c_lohi", "A.u_c_lolo",
        "A.u_i_hihi", "A.u_i_hilo", "A.u_i_lohi", "A.u_i_lolo",
        "A.u_s_hihi", "A.u_s_hilo", "A.u_s_lohi", "A.u_s_lolo",
        "B.u_c_hihi", "B.u_c_hilo", "B.u_c_lohi", "B.u_c_lolo",
        "B.u_i_hihi", "B.u_i_hilo", "B.u_i_lohi", "B.u_i_lolo",
        "B.u_s_hihi", "B.u_s_hilo", "B.u_s_lohi", "B.u_s_lolo"
    ]
   
    run_all_datasets(datasets512, 512, 16, parse_etcs_512, title="512-Task Datasets")
    run_all_datasets(datasets1024, 1024, 32, parse_etcs_1024, title="1024-Task Datasets")
    run_all_datasets(datasets2048,2048, 64, parse_etcs_1024, title="2048-Task Datasets")
    run_all_datasets(datasets4096, 4096, 128, parse_etcs_1024, title="4096-Task Datasets")
    run_all_datasets(datasets8192, 8192, 256, parse_etcs_1024, title="8192-Task Datasets")
