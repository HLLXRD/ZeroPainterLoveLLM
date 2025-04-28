import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import time


class ExtractLLM:
    def __init__(self, seed=36):
        self.seed = seed if seed is not None else int(time.time())
        self.model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extracting(self, query, seed=36):
        # Seed handling
        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed)

        # Rest of your code remains unchanged
        prompt = f'''Explain the furniture description and clarify its components, following the given instruction.
        **Instruction:**
            - Only response in one line.
            - The response must not put inside a quotation mark.
            - The word must be suitable for kids.
            - Just simplify the description.
            - DO NOT include any other information.
            - Ignore the information about the background, for example "white background".
            - If there is geometrical or shape information, keep it.
            - Replace "a large cabinet" with "a half black half white cabinet".
            - If the number of doors or objects is mentiened, keep it.
            - DO NOT use 'a' or 'an' at the beginning of the sentence.
        **Example:**
            Input: "A cylinder toilet tissue with a pink rabbit pattern."
            Response: A cylinder-shaped toilet tissue with pink rabbit pattern. 

            Input: "A large cabinet with four tall doors in the middle, providing ample storage space for various items."
            Response: A half black half white cabinet with four doors in the middle, providing ample storage space for various items.

        **Input:** "{query}"
'''  # Your prompt template
        messages = [
            {"role": "system", "content": "You are a teacher. Always use simple, common word"},
            {"role": "user", "content": prompt}
        ]  # Your message formatting

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        output = self.model.generate(
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=True,  # Now seed-controlled
            temperature=0.1,
            top_p=0.9,
            num_return_sequences=1
        )
        raw = self.tokenizer.decode(output[0], skip_special_tokens=True)

        def extract_response(raw):
            """Extract only the response part from the complete output"""
            # Look for patterns like "Output:" or the assistant's response after the prompt
            raw = raw.split("[|assistant|]")
            extracted_raw = raw[1]
            return extracted_raw

        final = extract_response(raw)
        print(final)
        txt_path = os.path.join("/root/ZeroPainter", 'llmkindergarten.txt')
        with open(txt_path, 'a') as f:
            f.write(query + "------>" + final + '\n')
        return final


if __name__ == "__main__":
    # Usage
    hlong = ExtractLLM()

    hlong.extracting(
        "a large cabinet with two tall cabinets on either side of four tall doors in the middle, providing ample storage space for various items.")

# This will get 9.54 on the pure pano2makatmost, mainatmost