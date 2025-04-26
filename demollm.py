import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import json


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
        prompt = f'''Explain this furniture description and clarify its components:"{query}"
        Instruction:
            - Only response in one line.
            - The response must not put inside a quotation mark.
            - The word must be suitable for kids.
            - Just simplify the description.
            - Dont't add more information to it.
            - Ignore the information about the background, for example "white background".
            - Keep "cylindrical", "rectangular", "triangle"
        Example:
            User: "A cylindrical toilet tissue with a pink rabbit pattern."
            Response: A cylindrical-shaped toilet tissue with pink rabbit pattern. 
'''

        messages = [
            {"role": "system", "content": "You are a teacher. Always use simple, common word"},
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
        txt_path = os.path.join("/root/ZeroPainter", 'llmteacher.txt')
        with open(txt_path, 'a') as f:
            f.write(query + "------>" + final + '\n')
        return final

    def list_extracting(self, query):
        prompt_extract = f'''Extract important information from this text: "{query}"
        Instruction:
            - The result must be in a list of string, for example: ["information1", "information2", "information3"].
            - Ignore all the information about the usage of the object.
            - Remove all the "background" information.
            - Keep the core furniture object.
        Example:
            User: "A glass vase with delicate pattern to arrange flower on it."
            Response: ["glass vase", "delicate pattern"]

            User: "An iron locket with some decorative stripes for clothes storage in a white background."
            Response: ["iron locket", "decorative stripes"]

            User: "A oval-shaped chandelier featuring with black cubes around"
            Response: ["oval-shaped chandelier", "black cubes around"]

        '''
        messages_extract = [
            {"role": "system",
             "content": "You are an expert information extractor. Always use brief, simple , precise word"},
            {"role": "user", "content": prompt_extract}
        ]
        input_ids_extract = self.tokenizer.apply_chat_template(
            messages_extract,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        output_extract = self.model.generate(
            input_ids_extract,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=False,
        )

        raw = self.tokenizer.decode(output_extract[0], skip_special_tokens=True)

        def extract_response(raw):
            """Extract only the response part from the complete output"""
            # Look for patterns like "Output:" or the assistant's response after the prompt
            raw = raw.split("[|assistant|]")
            extracted_raw = raw[1]
            return extracted_raw

        final = extract_response(raw)
        element_list = json.loads(final)
        print(element_list)
        return element_list
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
    hlong = ExtractLLM()
    # Usage
    all_scene_path = "/root/private_data/scenes"
    for dir in os.listdir(all_scene_path):
        path_dir = os.path.abspath(dir)
        with open(os.path.join(all_scene_path, dir, "query.txt")) as f:
            query = f.read()
            result_list = f'{hlong.list_extracting(query)}'
            with open("/root/ZeroPainter/list_extracted.txt", 'a') as q:
                q.write(query + "-------->" + result_list + "\n")

    # hlong = ExtractLLM()

    # hlong.list_extracting(
    #     "a wooden armoire featuring two doors on either side and a central window or door adorned with floral-patterned curtains. The armoire's top is embellished with decorative carvings, while its bottom boasts four legs. Against a white background, the armoire is prominently showcased as if in an advertisement.")

# # Test case 2
# hlong.extracting("a modern ceiling light fixture featuring multiple glass orbs suspended from a black metal bar, likely intended to provide ambient lighting for a room or space.")


# # Test case 3
# hlong.extracting("a tall, dark brown shelving unit with open cubbies for hanging clothes.")


# # Test case 4
# hlong.extracting("a toy shelf containing various objects such as fruits, toys, and an owl, showcasing a colorful arrangement of items on its shelves.")


# Threshold 2200