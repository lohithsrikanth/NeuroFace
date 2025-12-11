import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Hook for forward pass: save feature maps
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Hook for backward pass: save gradients
        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; we want grad wrt output
            self.gradients = grad_output[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        # For newer PyTorch versions, backward hook variant:
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: [1, 3, H, W] normalized tensor
        class_idx: if None, use predicted class
        returns: cam heatmap as [H, W] in [0,1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # [1, num_classes]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Scalar for backprop
        score = logits[0, class_idx]
        score.backward()

        # gradients: [1, C, H', W']
        # activations: [1, C, H', W']
        gradients = self.gradients       # d(score)/d(feature_maps)
        activations = self.activations

        # Global average pool over spatial dims to get channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted sum across channels
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU
        cam = F.relu(cam)

        # Normalize to [0,1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # [H, W]

        return cam, class_idx

def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def tensor_to_image(tensor):
    """
    Convert normalized tensor [3,H,W] back to uint8 image [H,W,3] in [0,255]
    """
    img = tensor.clone().cpu().numpy()
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)  # unnormalize
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def overlay_cam_on_image(img, cam, alpha=0.4):
    """
    img: [H,W,3] uint8
    cam: [H,W] float in [0,1]
    returns overlay image [H,W,3] uint8
    """
    h, w, _ = img.shape
    cam_color = plt.get_cmap("jet")(cam)[:, :, :3]  # [H,W,3], ignore alpha
    cam_color = (cam_color * 255).astype(np.uint8)

    overlay = (alpha * cam_color + (1 - alpha) * img).astype(np.uint8)
    return overlay

def load_model(checkpoint_path, device):
    model = get_resnet("resnet18", pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def visualize_grad_cam_on_image(
    img_path,
    checkpoint_path,
    class_names,
    #output_path="/content/drive/MyDrive/cs6073/NeuroFace/src/gradcam_output.png",
    output_path="gradcam_output.png",
    target_class_idx=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = load_model(checkpoint_path, device)

    # 2. Choose target layer (last conv layer of ResNet-18)
    # ResNet-18: model.layer4 is Sequential of BasicBlock; we pick the last block's conv2
    target_layer = model.layer4[-1].conv2

    # Instantiate the GradCAM class (create a GradCAM object)
    grad_cam = GradCAM(model, target_layer)
    # 3. Load and preprocess image
    preprocess = get_preprocess_transform()
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,224,224]

    # 4. Compute Grad-CAM
    cam, used_class_idx = grad_cam(input_tensor, class_idx=target_class_idx)

    grad_cam.remove_hooks()

    # 5. Convert back to image and overlay heatmap
    img_np = tensor_to_image(input_tensor[0])  # [H,W,3]
    overlay = overlay_cam_on_image(img_np, cam, alpha=0.4)

    # 6. Plot and save
    class_label = class_names[used_class_idx] if class_names is not None else str(used_class_idx)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title(f"Original ({class_label})")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Grad-CAM")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    #plt.close()

    print(f"Saved Grad-CAM visualization to {output_path}")
