# import sys
# import os

# # Get the directory where main.py is
# current_dir = os.path.dirname(os.path.abspath(_file_))
# # The project root is the parent of this directory
# project_root = os.path.abspath(os.path.join(current_dir))

# # Add the root to sys.path
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from src.zeropainter.zero_painter_pipline import ZeroPainter
# from src.zeropainter import models, dreamshaper, segmentation
# from src.zeropainter import zero_painter_dataset
# import torch
# from torchvision.utils import save_image
# import json


# def HLongBeo(mask_pil_png, metadata):
#     metadata = json.loads(metadata)
#     config_folder_for_models = 'config'
#     model_folder_inpiting = 'models/sd-1-5-inpainting'
#     model_folder_generation = 'models/sd-1-4'
#     segment_anything_model = 'models/sam_vit_h_4b8939.pth'

#     model_inp, _ = models.get_inpainting_model(config_folder_for_models, model_folder_inpiting)
#     model_t2i, _ = models.get_t2i_model(config_folder_for_models, model_folder_generation)
#     model_sam = segmentation.get_segmentation_model(segment_anything_model)
#     zero_painter_model = ZeroPainter(model_t2i, model_inp, model_sam)

#     data = zero_painter_dataset.ZeroPainterDataset(mask_pil_png, metadata)
#     name = mask_pil_png.split('/')[-1]
#     # Get the tensor output
#     result_tensor = zero_painter_model.gen_sample(data[0], 37, 37, 30, 30)  # Shape: (3, H, W)
#     # Save the raw tensor (preserves all data/metadata)
#     #torch.save(result_tensor.cpu(), "result_tensor.pt")
#     #print("Saved tensor to 'result_tensor.pt'")
#     # Normalize tensor to [0, 1] and save
#     #save_image(result_tensor.float() / 255.0, "result.png")
#     #print("Saved image to 'result.png'")
#     # Optional: Convert to numpy array for other uses
#     # result_np = result_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

#     return result_tensor  # Return as PyTorch tensor
import sys
import os

# Get the directory where main.py is
current_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is the parent of this directory
project_root = os.path.abspath(os.path.join(current_dir))

# Add the root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.zeropainter.zero_painter_pipline import ZeroPainter
from src.zeropainter import models, dreamshaper, segmentation
from src.zeropainter import zero_painter_dataset
import torch
from torchvision.utils import save_image
import json
import numpy as np
import sys
import os
import cv2
from torchvision.transforms.functional import to_pil_image
from demollm import ExtractLLM

# Now you can import from modules in the parent folder as needed.
class HLongBeo:
    def __init__(self):
        zeropainter_root = os.path.dirname(os.path.abspath(__file__))
        self.config_folder_for_models = os.path.join(zeropainter_root, "config")
        self.model_folder_inpiting = os.path.join(zeropainter_root, "models/sd-1-5-inpainting")
        self.model_folder_generation = os.path.join(zeropainter_root, "models/sd-1-4")  # <-- THIS LINE
        self.segment_anything_model = os.path.join(zeropainter_root, "models/sam_vit_h_4b8939.pth")
        model_inp, _ = models.get_inpainting_model(self.config_folder_for_models, self.model_folder_inpiting)
        model_t2i, _ = models.get_t2i_model(self.config_folder_for_models, self.model_folder_generation)
        model_sam = segmentation.get_segmentation_model(self.segment_anything_model)
        self.zero_painter_model = ZeroPainter(model_t2i, model_inp, model_sam)
        self.extract_model = ExtractLLM()

    def ZP(self, mask_pil_png, text):
        extracted_text = self.extract_model.extracting(text)
        txt_path = os.path.join("/root/ZeroPainter", 'llm.txt')
        with open(txt_path, 'a') as f:
            f.write(str(extracted_text) + '\n')
        # __file__ is the current file's path (assumed to be inside the ZeroPainter directory)
        # Get the absolute path of the current file
        current_file_path = os.path.abspath(__file__)
        # Get the directory containing the current file (ZeroPainter folder)
        zero_painter_dir = os.path.dirname(current_file_path)
        # Get the direct parent folder of ZeroPainter
        parent_folder = os.path.dirname(zero_painter_dir)
        # Insert the parent folder into sys.path if it's not already there
        if parent_folder not in sys.path:
            sys.path.insert(0, parent_folder)

        # Now you can import from modules in the parent folder as needed.
        img = np.array(mask_pil_png)
        # # Load ảnh (BGR)
        # # print(mask_pil_png)
        # img = cv2.imread(mask_pil_png)

        # Tạo mặt nạ vùng KHÔNG PHẢI trắng (dung sai nhỏ để xử lý noise/mờ)
        tolerance = 20
        non_white_mask = ~np.all(img > (255 - tolerance), axis=-1)

        # Chuẩn hóa vùng không trắng về 1 màu duy nhất, ví dụ đỏ
        img[non_white_mask] = [235, 206, 135]

        # Bây giờ chỉ còn một màu duy nhất đại diện cho vùng mark, có thể dùng unique hoặc mode để lấy RGB
        unique_colors = np.unique(img.reshape(-1, 3), axis=0)
        cv2.imwrite("output.png", img)

        # print(unique_colors)
        mask_pil_png = img
        metadata = [{
    "prompt": extracted_text + "in a completely solid white background (255,255,255) RGB with no objects around it.",
    "color_context_dict": {
        "(235, 206, 135)": extracted_text
    }
}]
# "{(21, 0, 136)}": "black metal bar"
# "{(164, 73, 163)}": "doors",
# "{(231, 191, 200)}": "doors",
# "{(204, 72, 63)}": "doors"
        #metadata = json.loads(metadata)
        # Absolute paths using zeropainter_root

        data = zero_painter_dataset.ZeroPainterDataset(mask_pil_png, metadata)
        # name = mask_pil_png.split('/')[-1]
        # Get the tensor output
        result_tensor = self.zero_painter_model.gen_sample(data[0], 37, 37, 30, 30)  # Shape: (3, H, W)
        # Save the raw tensor (preserves all data/metadata)
        # torch.save(result_tensor.cpu(), "result_tensor.pt")
        # print("Saved tensor to 'result_tensor.pt'")
        # Normalize tensor to [0, 1] and save
        save_image(result_tensor.float() / 255.0, "resultnew.png")
        # print("Saved image to 'result.png'")
        # Optional: Convert to numpy array for other uses
        # result_np = result_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        # Add new dim
        result_tensor = (result_tensor.float() / 255.0).cpu()
        result_tensor_pil = to_pil_image(result_tensor)

        return result_tensor_pil  # Return as PyTorch tensor

if __name__ == "__main__":
    hlong = HLongBeo()
    hlong.ZP("/root/c487f433-4414-4966-a275-6dfade36d2f0_mask.png",
            "a sleek black light fixture with six glass bulbs arranged in a symmetrical pattern on a white background.")