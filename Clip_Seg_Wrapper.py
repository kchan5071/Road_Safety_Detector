from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import Smoothness_Detector

class Clip_Seg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def turn_preds_to_images(self, preds, image):
        image_np = np.array(image)
        segmented_images = []
        for i in range(len(preds)):
            heatmap = torch.sigmoid(preds[i][0])
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)

            w, h = image_np.shape[0], image_np.shape[1]
            heatmap_resized = F.interpolate(
                heatmap,
                size=(w, h),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            threshold = 0.45
            binary_mask = (heatmap_resized > threshold).astype(np.uint8)

            segmented_image = np.zeros_like(image_np)
            segmented_image[binary_mask == 1] = image_np[binary_mask == 1]

            segmented_images.append(segmented_image)
        return segmented_images

    def segment_image(self, image, prompts):
        inputs = self.processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = (outputs.logits/1).unsqueeze(1)
        return self.turn_preds_to_images(preds, image)[0]
        


def main():
    clip_seg = Clip_Seg()
    image_names = ["rd_test_1.jpg", "rd_test_2.jpg"]
    prompts = ["road", "road"]
    alpha_low = 4
    alpha_high = 200
    for image_name in image_names:
        image = Image.open(image_name)
        segmented_image = clip_seg.segment_image(image, prompts)
        gray_image = Smoothness_Detector.convert_to_gray(segmented_image)
        alpha_filtered_image_low = Smoothness_Detector.alpha_filter_low(gray_image, alpha_low)
        alpha_filtered_image = Smoothness_Detector.alpha_filter_high(alpha_filtered_image_low, alpha_high)
        smoothness = Smoothness_Detector.get_smoothness(alpha_filtered_image)
        print(smoothness)
        plt.imshow(alpha_filtered_image, cmap='gray')
        plt.show()



if __name__ == "__main__":
    main()