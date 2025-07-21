from setproctitle import setproctitle
setproctitle("국선호한국어")

path = "한국어_cma"
import sys
import json
with open(sys.argv[1], "r") as f:
   t_list = json.load(f)[f"{path}_t_list"]
import os   
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # 한국어
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evomerge.models.causallm import CausalLMWithTransformers
import copy
import gc
from dataclasses import asdict
from evomerge import instantiate_from_config, load_config, set_seed
import warnings
import logging
logging.getLogger().setLevel(logging.ERROR)
from transformers import logging as hf_logging
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger().setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"



# 모델 불러오기
model_A = CausalLMWithTransformers(
    "davidkim205/komt-mistral-7b-v1",
    model_kwargs={"torch_dtype": torch.float16},
    template='ko-alpaca-cot'
)


model_B = CausalLMWithTransformers(
    "WizardLM/WizardMath-7B-V1.1",
    model_kwargs={"torch_dtype": torch.float16},
    template='ko-alpaca-cot'
)




state_A = model_A.model.state_dict()
state_B = model_B.model.state_dict()




merge_method = "slerp"

merged_state = copy.deepcopy(state_A)



layer_names = list(merged_state.keys())
num_layers = len(layer_names)

base_weight = 0.4
amplitude = 0.2



# 병합 과정
for idx, k in enumerate(merged_state):
    layer_idx = min(idx, len(t_list) - 1)
    if k in state_B:

        if merge_method == "slerp":
            t = t_list[layer_idx]
            merged_state[k] = t * state_A[k] + (1 - t) * state_B[k]
            
        
        if merge_method == "slerp_sin":
            a = state_A[k]
            b = state_B[k]
            t = t_list[layer_idx]

            a_flat = a.view(-1)
            b_flat = b.view(-1)

            

            a_norm = a_flat / a_flat.norm()
            b_norm = b_flat / b_flat.norm()

            cos_theta = (a_norm * b_norm).sum().clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)


            if theta.abs() < 1e-5:
                merged = (1 - t) * a + t * b
            else:
                sin_theta = torch.sin(theta)
                w1 = torch.sin((1 - t) * theta) / sin_theta
                w2 = torch.sin(t * theta) / sin_theta
                merged = w1 * a + w2 * b

            merged_state[k] = merged




model_A.model.load_state_dict(merged_state)
tokenizer = model_A.tokenizer

save_path = f"./merged_model/{path}"

model_A.model.save_pretrained(save_path)

tokenizer.save_pretrained(save_path)


    

# 평가
config_path = f"./configs/{path}.yaml"
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

entry = {"t": t, "acc": acc, "acc_ko": acc_ko}
print(acc_ko, flush=True)



