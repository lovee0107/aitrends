from setproctitle import setproctitle
setproctitle("국선호일본어")
import torch
from pathlib import Path
import os, json, subprocess
import cma
import numpy as np
from contextlib import redirect_stdout

good = 0.0
idx = 1

path = "일본어_cma"


def evaluate_t_list(t_list):
    global good, path, idx

    os.makedirs("tmp", exist_ok=True)
    with open(f"tmp/{path}_t_list.json", "w") as f:
        json.dump({f"{path}_t_list": t_list.tolist()}, f) 
    result = subprocess.run(
        ["python3", f"./{path}.py", f"tmp/{path}_t_list.json"],
        capture_output=True, text=True
    )

    lines = result.stdout.strip().splitlines()
    last_line = lines[-1] if lines else ""




    acc = float(last_line)
    
    print(f"평가 완료: acc_ja = {acc}", flush=True)

    with open(f"results/일본어/{path}_history_log.txt", "a") as log:
        log.write(f"{es.countiter}, acc_ja = {acc}\n")

    last_line = float(last_line)
    if good <= last_line:
        good = last_line
        print("저장하기")
        with open(f"results/일본어{path}_t_list_good_{good}.json", "w") as f:
            json.dump({f"{path}_t_list": t_list.tolist()}, f, indent=4)

    cov_matrix = es.sm.C.copy()
    mean_vector = es.mean.copy()



    if idx % 21 == 0:
        print(np.diag(cov_matrix))
        print(mean_vector)
        print(es.countiter)

    idx = idx + 1
    return -acc
    




# CMA-ES 초기화
layer_count = 291
es = cma.CMAEvolutionStrategy([0.5] * layer_count, 0.1)

os.makedirs("results/일본어", exist_ok=True)
with open(f"results/일본어/{path}_cma_stdout.log", "w") as f_out:
    with redirect_stdout(f_out):
        es.optimize(evaluate_t_list, iterations=100)


best_t, best_loss,uuu = es.best.get()
best_acc = -best_loss


print("최적 t_list =", best_t)
print("최고 acc_ja =", best_acc)

# 최종 결과 저장
best_t = es.result.xbest.tolist()
os.makedirs("results/일본어", exist_ok=True)
with open(f"results/일본어/{path}_cma_best_t_list.json", "w") as f:
    json.dump({f"{path}_t_list": best_t}, f, indent=2)

print("최적 t_list =", best_t)