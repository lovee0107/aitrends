from setproctitle import setproctitle
setproctitle("국선호")
import os
import gc
import json
import logging
from dataclasses import asdict
from evomerge import instantiate_from_config, load_config, set_seed

# 2. 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 3. config 경로 설정 (✅ 여기를 너 config yaml 파일 경로로 바꿔줘)
config_path = f"./configs/중국어/Merged-Traditional-Chinese-WizardMath.yaml"  # 예시
output_path = None  # 자동 생성

# 4. output_path 없으면 자동 생성
if output_path is None:
    output_path = (
        os.path.splitext(os.path.basename(config_path))[0] + ".json"
    )
    output_path = f"results/중국어/{output_path}"
    os.makedirs("results", exist_ok=True)

assert output_path.endswith(".json"), "`output_path` must be json file"


# 6. config 파일 읽기
config = load_config(config_path)

# 7. seed 고정
set_seed(42)

# 8. 모델 로드 (자동으로 GPU로 올림)
model = instantiate_from_config(config["model"])

# 9. 평가 준비
eval_configs = config["eval"]
if isinstance(eval_configs, dict):
    eval_configs = [eval_configs]

results = {}

# 10. 평가 실행
for eval_config in eval_configs:
    evaluator = instantiate_from_config(eval_config)
    
    outputs = evaluator(model)
    results[evaluator.name] = asdict(outputs)

acc = outputs.metrics["acc"]
acc_zh = outputs.metrics["acc_zh"]

entry = {"acc": acc, "acc_zh": acc_zh}
print(entry, flush=True)
