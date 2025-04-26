import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os


class ExtractLLM:
    def __init__(self):
        self.model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extracting(self, query):
        prompt = f'''Analyze this furniture description and summarize its components:"{query}"

        Instruction:
            - The response must not be put inside quotation marks "".
            - Ignore the word "symmetrically".
            - Only summarize it on one line.
            - Ignore the word "flat".
            - Remove all descriptions about the usage of the objects. 
            - Keep the core furniture words and especially the words for color, texture, material of it. 
            - Don't remove any numerals or quantifiers word like "two", "four", "six", "multiple".
            - Please remove all information about background color, such as "white background".
            - Make the words as simple as possible. For example, the word "fixture with" can be shortened to "with".
            - Remove the number on the appearance.
            - If there is shape-related words, add "-shaped" before the word, for example "cylindrical" change to "cylindrical-shaped".
'''

        messages = [
            {"role": "system", "content": "You are an expert simplifier. Always use plain, concise language."},
            {"role": "user", "content": prompt}
        ]

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
            do_sample=False,
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
        txt_path = os.path.join("/root/ZeroPainter", 'llm.txt')
        with open(txt_path, 'a') as f:
            f.write(query + "------>" + final + '\n')
        return final
        # print("\n ====END OF RESPONSE====\n")

        # # Improved parsing
        # main_object = None
        # auxiliary_items = []

        # # Extract main object
        # main_match = re.search(r'Main Object:\s*(.+?)(\n|$)', raw)
        # if main_match:
        #     main_object = main_match.group(1).strip()

        # # Extract auxiliary items
        # aux_block = re.search(r'Auxiliary Items:\s*(.+?)(\n\n|$)', raw, re.DOTALL)
        # if aux_block:
        #     items = re.findall(r'\d+\.\s*(.+)|-\s*(.+)', aux_block.group(1))
        #     for item in items:
        #         clean_item = (item[0] or item[1]).strip()
        #         if clean_item.lower() != 'none':
        #             auxiliary_items.append(clean_item)

        # return main_object, auxiliary_items


if __name__ == "__main__":
    # Usage
    hlong = ExtractLLM()

    hlong.extracting(
        "a modern coffee table featuring two round, black tables that can be joined together or separated to form a single unit, comprising circular shapes made of glossy black material with a reflective surface, supported by a central base.")

# # Test case 2
# hlong.extracting("a modern ceiling light fixture featuring multiple glass orbs suspended from a black metal bar, likely intended to provide ambient lighting for a room or space.")


# # Test case 3
# hlong.extracting("a tall, dark brown shelving unit with open cubbies for hanging clothes.")


# # Test case 4
# hlong.extracting("a toy shelf containing various objects such as fruits, toys, and an owl, showcasing a colorful arrangement of items on its shelves.")