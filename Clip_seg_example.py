from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
# image = Image.open('/content/00000007.jpg')
image = Image.open('rd_test.jpg')

prompts = ["road", "road"]
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**inputs)
preds = (outputs.logits/1).unsqueeze(1)

_, ax = plt.subplots(1, len(prompts) + 1, figsize=(3*(len(prompts) + 1), 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
ax[1].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
[ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)];

image_np = np.array(image)

# Process each prompt
prompts = ["road"]
segmented_images = []

for i, prompt in enumerate(prompts):
    # Generate heatmap for each prompt
    heatmap = torch.sigmoid(preds[i][0])
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    # Resize heatmap to match original image dimensions
    w, h = image_np.shape[0], image_np.shape[1]
    heatmap_resized = F.interpolate(
        heatmap,
        size=(w, h),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Apply threshold to create a binary mask
    threshold = 0.45
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Create segmented image for the current prompt
    segmented_image = np.zeros_like(image_np)
    segmented_image[binary_mask == 1] = image_np[binary_mask == 1]

    # Store each segmented image for display
    segmented_images.append(segmented_image)

# Plot the original image and each segmented image for each prompt
_, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, 5))
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")

for i, segmented_image in enumerate(segmented_images):
    ax[i + 1].imshow(segmented_image)
    ax[i + 1].set_title(f"Segmented Image - {prompts[i]}")
    ax[i + 1].axis("off")

plt.show()

print(segmented_image.shape)


# Assuming 'image_np' is the numpy array of the original image
# Assuming 'preds' is the output from the model, containing logits for all prompts

# Get image dimensions
w, h = image_np.shape[:2]

# Create a figure to display the original and segmented images
num_prompts = len(prompts)
_, ax = plt.subplots(1, num_prompts + 1, figsize=(3 * (num_prompts + 1), 4))

# Display the original image
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")
segmented_images = []
# Process each prompt
for i in range(num_prompts):
    # Get the heatmap for the current prompt
    heatmap = torch.sigmoid(preds[i][0])
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    # Resize the heatmap to the original image size
    heatmap_resized = F.interpolate(
        heatmap,
        size=(w, h),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Apply threshold to create a binary mask
    threshold = 0.6
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Create a segmented image based on the binary mask
    segmented_image = np.zeros_like(image_np)
    segmented_image[binary_mask == 1] = image_np[binary_mask == 1]
    segmented_images.append(segmented_image)
    # Display the segmented image for the current prompt
    ax[i + 1].imshow(segmented_image)
    ax[i + 1].set_title(prompts[i])
    ax[i + 1].axis("off")

plt.tight_layout()
plt.show()

# Assuming 'image_np' is the numpy array of the original image
# Assuming 'preds' is the output from the model, containing logits for all prompts

# Get image dimensions
w, h = image_np.shape[:2]

# Create a figure to display the original and segmented images
num_prompts = len(prompts)
_, ax = plt.subplots(1, num_prompts + 1, figsize=(3 * (num_prompts + 1), 4))

# Display the original image
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")
segmented_images = []
# Process each prompt
for i in range(num_prompts):
    # Get the heatmap for the current prompt
    heatmap = torch.sigmoid(preds[i][0])
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    # Resize the heatmap to the original image size
    heatmap_resized = F.interpolate(
        heatmap,
        size=(w, h),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Apply threshold to create a binary mask
    threshold = 0.6
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Create a segmented image based on the binary mask
    segmented_image = np.zeros_like(image_np)
    segmented_image[binary_mask == 1] = image_np[binary_mask == 1]
    segmented_images.append(segmented_image)
    # Display the segmented image for the current prompt
    ax[i + 1].imshow(segmented_image)
    ax[i + 1].set_title(prompts[i])
    ax[i + 1].axis("off")

plt.tight_layout()
plt.show()


# Assuming 'image_np' is the numpy array of the original image
# Assuming 'preds' is the output from the model, containing logits for all prompts

# Get image dimensions
w, h = image_np.shape[:2]

# Create a figure to display the original and segmented images
num_prompts = len(prompts)
_, ax = plt.subplots(1, num_prompts + 1, figsize=(3 * (num_prompts + 1), 4))

# Display the original image
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")
segmented_images = []
# Process each prompt
for i in range(num_prompts):
    # Get the heatmap for the current prompt
    heatmap = torch.sigmoid(preds[i][0])
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    # Resize the heatmap to the original image size
    heatmap_resized = F.interpolate(
        heatmap,
        size=(w, h),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Apply threshold to create a binary mask
    threshold = 0.6
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)

    # Create a segmented image based on the binary mask
    segmented_image = np.zeros_like(image_np)
    segmented_image[binary_mask == 1] = image_np[binary_mask == 1]
    segmented_images.append(segmented_image)
    # Display the segmented image for the current prompt
    ax[i + 1].imshow(segmented_image)
    ax[i + 1].set_title(prompts[i])
    ax[i + 1].axis("off")

plt.tight_layout()
plt.show()