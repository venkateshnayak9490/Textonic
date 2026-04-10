from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenLLMClient:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

    def generate(self, prompt=None, user_prompt=None, system_prompt=None, max_tokens=200):
        #  Handle both calling styles used in your code
        if user_prompt:
            prompt = user_prompt

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean response
        if "assistant" in response:
            response = response.split("assistant")[-1]

        return response.strip()