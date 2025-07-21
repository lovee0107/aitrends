import os
import logging
from typing import List, Union, Optional
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from vllm import LLM, SamplingParams

from .prompt_templates import JA_ALPACA_COT_TEMPLATE
from .prompt_templates import KO_ALPACA_COT_TEMPLATE
from .prompt_templates import ZH_ALPACA_COT_TEMPLATE
from .utils import set_template, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class StopOnWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, device):
        super().__init__()
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.device = device
        self.stop_word_ids = [
            tokenizer.encode(word, add_special_tokens=False) for word in stop_words
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # input_ids: (batch_size, seq_len)
        for stop_seq in self.stop_word_ids:
            if len(stop_seq) == 0:
                continue
            if (input_ids[0, -len(stop_seq):] == torch.tensor(stop_seq, device=self.device)).all():
                return True
        return False


class kkkCausalLMWithTransformers:
    def __init__(self, model_path, model_kwargs=None, template=None):
        #self.device = "cuda:7" if not torch.cuda.is_available() else "cpu"
        #print(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **(model_kwargs or {})
        )
        self.template = template

        # 🔥 stop tokens 설정
        self.stop_words = ["Instruction:", "Instruction", "Response:", "Response"]

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            stopping_criteria = StoppingCriteriaList([
                StopOnWordsCriteria(self.stop_words, self.tokenizer, self.device)
            ])

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                temperature=0,
                top_p=1.0,
                repetition_penalty=1.0,
                stopping_criteria=stopping_criteria,  # 🔥 vLLM처럼 stop word에서 멈추기
                eos_token_id=self.tokenizer.eos_token_id,  # 🔥 만약 아무 stop word도 없으면 eos로 멈추기
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses

    def __call__(self, text, **kwargs):
        return self.generate(text, **kwargs)


class CausalLMWithTransformers:
    def __init__(self, model_path, model_kwargs=None, template=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **(model_kwargs or {})
        )

        # 🔧 device_map을 쓰는 경우 to(self.device) 금지
        # device 설정은 generate 내부에서 input에 따라 처리

        self.template = template

        # 🔥 stop tokens 설정
        self.stop_words = ["Instruction:", "Instruction", "Response:", "Response","\n\n", "###", "답:", "Answer:"]

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"🧠 문제 {i+1} 풀고 있음...") 
            # 🔧 device는 모델 weight 기준으로 자동 설정
            device = self.model.device

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)  # ⬅️ 여기서만 device 사용

            stopping_criteria = StoppingCriteriaList([
                StopOnWordsCriteria(self.stop_words, self.tokenizer, device)
            ])

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                temperature=0,
                top_p=1.0,
                repetition_penalty=1.0,
                stopping_criteria=stopping_criteria,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses

    def __call__(self, text, **kwargs):
        return self.generate(text, **kwargs)

class CausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    default_template = JA_ALPACA_COT_TEMPLATE

    def __init__(
        self,
        model_path: str = None,
        template: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        self.model = LLM(model=model_path, **self.model_kwargs)
        self.post_init()

    def post_init(self):
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text




class KoCausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    default_template = KO_ALPACA_COT_TEMPLATE

    def __init__(
        self,
        model_path: str = None,
        template: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        print("model_kwargs:", self.model_kwargs)
        self.model = LLM(model=model_path, **self.model_kwargs)
        self.post_init()

    def post_init(self):
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        #print('hello')
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text


class ZhCausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.85,
        "repetition_penalty": 1.5,
    }
    default_template = ZH_ALPACA_COT_TEMPLATE

    def __init__(
        self,
        model_path: str = None,
        template: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        print("model_kwargs:", self.model_kwargs)
        self.model = LLM(model=model_path, **self.model_kwargs)
        self.post_init()

    def post_init(self):
        stop_tokens = ["\n\n", "###", "答：", "回答：", "答案：", "Answer:"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        #print('hello')
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text