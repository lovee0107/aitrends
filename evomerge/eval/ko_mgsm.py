from typing import Any, Dict, List, Optional
import re

from datasets import load_dataset

from .metrics import LanguageDetector, mean
from .utils import EvalOutput


def extract_answer_number(completion: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))


class KoMGSM:
    name = "MGSM-KO"
    dataset_path = "freewheelin/mgsm_ko"
    dataset_split = "test"

    def __init__(
        self,
        verbose: bool = False,
        strict_korean: bool = True,
    ):
        self.verbose = verbose
        self.collate_fn = None
        dataset = load_dataset(
            path=self.dataset_path, split=self.dataset_split
        )
        self.dataset = dataset.select_columns(["question", "answer_number"])
        self.dataset = dataset.select(range(250))
        self.lang_detect = None
        if strict_korean:
            self.lang_detect = LanguageDetector()

    def compute_score(self, results: Dict[str, List[Any]]) -> Dict[str, float]:
        corrects = [
            answer == pred
            for answer, pred in zip(results["answer"], results["prediction"])
        ]
        res_dict = {"acc": mean(corrects)}
        # detect Japanese by fasttext and replace empty string if it's not Ja
        if self.lang_detect:
            corrects_ko = []
            for i, resp in enumerate(results["response"]):
                res = self.lang_detect(resp)
                corrects_ko.append(corrects[i] and res.get("__label__ko", 0.0) > 0.5)
            res_dict["acc_ko"] = mean(corrects_ko)
        return res_dict

    def __call__(self, model, **kwargs) -> EvalOutput:

        results = []
        question = [example["question"] for example in self.dataset]
        answers = [example["answer_number"] for example in self.dataset]
        # inference
        resps = model(text=question, **kwargs)
        print(f"[DEBUG] 질문 개수: {len(question)}, 모델 응답 개수: {len(resps)}")
        preds = [extract_answer_number(t) for t in resps]
        results = {
            "question": question,
            "answer": answers,
            "response": resps,
            "prediction": preds,
        }
        metrics = self.compute_score(results)

        return EvalOutput(metrics=metrics, results=results)
