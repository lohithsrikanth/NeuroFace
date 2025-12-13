
checkpoint_path = "/content/drive/MyDrive/cs6073/NeuroFace/src/resnet18_neuroface_best_model.pth"  # your trained model
img_path = "/content/drive/MyDrive/cs6073/NeuroFace/processed_frames/test/ALS/A011/A011_02_BBP_NORMAL_color.avi/424.jpg"   # one sample
class_names = ['ALS', 'HC', 'PS']       # match your folder order

# Option 1: Let Grad-CAM use the model's predicted class
visualize_grad_cam_on_image(
        img_path=img_path,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        output_path="/content/drive/MyDrive/cs6073/NeuroFace/src/gradcam_als_pred.png",
        target_class_idx=None
    )
