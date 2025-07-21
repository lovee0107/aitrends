from setproctitle import setproctitle
setproctitle("국선호")
import os   
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # 한국어
import torch
import gc
import json
import logging
from dataclasses import asdict
from evomerge import instantiate_from_config, load_config, set_seed

config_path = f"./configs/한국어/Merged-Komt-WizardMath.yaml"
output_path = None

if output_path is None:
    output_path = (
        os.path.splitext(os.path.basename(config_path))[0] + ".json"
    )
    output_path = f"results/한국어/{output_path}"
    os.makedirs("results", exist_ok=True)

assert output_path.endswith(".json"), "`output_path` must be json file"


config = load_config(config_path)

set_seed(42)

model = instantiate_from_config(config["model"])

eval_configs = config["eval"]
if isinstance(eval_configs, dict):
    eval_configs = [eval_configs]

results = {}

for eval_config in eval_configs:
    evaluator = instantiate_from_config(eval_config)
    
    outputs = evaluator(model)
    results[evaluator.name] = asdict(outputs)

acc = outputs.metrics["acc"]
acc_ko = outputs.metrics["acc_ko"]

entry = {"acc": acc, "acc_ko": acc_ko}
print(entry, flush=True)



