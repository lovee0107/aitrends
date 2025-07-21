from datasets import load_dataset
from .metrics import mean
from .utils import EvalOutput
import re
from typing import Any, Dict, List, Optional

def extract_answer_number(completion: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))

class GSM8K:
    name = "GSM8K"
    dataset_path = "gsm8k"
    dataset_split = "test"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.collate_fn = None
        dataset = load_dataset(self.dataset_path, name="main", split=self.dataset_split)
        self.dataset = dataset.select_columns(["question", "answer"])

    def compute_score(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        corrects = [
            extract_answer_number(answer) == pred
            for answer, pred in zip(results["answer"], results["prediction"])
            if pred is not None and extract_answer_number(answer) is not None
        ]
        return {"acc": mean(corrects)}

    def __call__(self, model) -> EvalOutput:
        dataset = self.dataset.select(range(10))  # ✅ 문제 수 줄이기 (올바른 방식)

        question = [example["question"] for example in dataset]
        answers = [example["answer"] for example in dataset]
        resps = model(text=question)
        preds = [extract_answer_number(r) for r in resps]

        return EvalOutput(
            metrics=self.compute_score({
                "answer": answers,
                "response": resps,
                "prediction": preds,
            }),
            results={
                "question": question,
                "answer": answers,
                "response": resps,
                "prediction": preds,
            }
        )