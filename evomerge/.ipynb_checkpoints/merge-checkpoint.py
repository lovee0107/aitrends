# evomerge/merge.py
import os, torch, copy
from .models.causallm import CausalLMWithTransformers

def merge_models(config):
    model_paths = [m["model_path"] for m in config["population"]]
    output_path = config["output_path"]
    method = config["merge_config"].get("merge_method", "slerp")


    print(f"🚀 Merging models: {model_paths} with method: {method}")

    base = CausalLMWithTransformers(
        model_path=model_paths[0],
        template="ja-alpaca-cot",
        model_kwargs={"torch_dtype": torch.float16}  # 👈 추가!
    )
    others = [
        CausalLMWithTransformers(
            model_path=p,
            template="ja-alpaca-cot",
            model_kwargs={"torch_dtype": torch.float16}  # 👈 추가!
        )
        for p in model_paths[1:]
    ]
    
    base_state = base.model.state_dict()
    other_states = [m.model.state_dict() for m in others]


    merged_state = copy.deepcopy(base_state)
    print(merged_state.shape())
    print("======================")
    for k in merged_state:
        #print(k)
        print("======================")
        if all(k in o for o in other_states):
            for o in other_states:
                #print(o)
                print("======================")
                if method == "linear":
                    #merged_state[k] += o[k]
                    pass
                elif method == "slerp":
                    #merged_state[k] = 0.5 * merged_state[k] + 0.5 * o[k]
                    pass
            if method == "linear":
                #merged_state[k] /= (len(other_states) + 1)
                pass
            
    print(merged_state["model.embed_tokens.weight"])
    exit()
    os.makedirs(output_path, exist_ok=True)
    torch.save(merged_state, os.path.join(output_path, "pytorch_model.bin"))
    print(f"[✓] Merged model saved to {output_path}")