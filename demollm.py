import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


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
            - Keep the core furniture words and especially the words for color, texture, material of it. 
            - Don't remove any numerals or quantifiers word like "two", "four", "multiple".
            - Please remove all information about background color, such as "white background".
            - Change the "featuring" to "with".
            - Make the words as simple as possible. For example, the word "fixture with" can be shortened to "with".

        Below are examples that you MUST follow, if the input is the same as one of the situation, you must output as the corresponding output:
            *Example 1:
                Input: "a sleek black light fixture with six glass bulbs arranged in a symmetrical pattern on a white background."
                Output: "black light fixture with six bulbs."
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

    # Test case 1
    hlong.extracting(
        "a wooden dresser featuring two doors on either side of a central windowed door adorned with floral-patterned curtains. The dresser is crafted from light-colored wood and boasts intricate gold accents along the edges of its doors and at the bottom of its central panel, adding a touch of elegance to its design.")

# # Test case 2
# hlong.extracting("a modern ceiling light fixture featuring multiple glass orbs suspended from a black metal bar, likely intended to provide ambient lighting for a room or space.")


# # Test case 3
# hlong.extracting("a tall, dark brown shelving unit with open cubbies for hanging clothes.")


# # Test case 4
# hlong.extracting("a toy shelf containing various objects such as fruits, toys, and an owl, showcasing a colorful arrangement of items on its shelves.")