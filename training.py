import torch
from ultralytics import YOLO

# Define a custom loss function
def custom_loss(pred, target, lambda_fp=2.0):
    # Assuming a simple example where we have pred and target as tensors
    # This is highly simplified and you would need to adapt it to your specific needs
    loss_fp = torch.sum((pred == 1) & (target == 0))  # False Positives
    loss_fn = torch.sum((pred == 0) & (target == 1))  # False Negatives
    loss_tp = torch.sum((pred == 1) & (target == 1))  # True Positives
    return lambda_fp * loss_fp + loss_fn + loss_tp

# Train the model with the custom loss function
def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Replace default loss function with custom loss function (if supported by YOLOv8)
    # Assuming model.loss_fn is the attribute to set custom loss function
    model.loss_fn = custom_loss  # This step assumes the library allows for setting custom loss directly

    # Train the model
    model.train(
        data='D:/Chris/Entrenamiento/dataset/data.yaml',
        epochs=100,
        imgsz=640,
        device=device,
        batch=-1,  # Adjust based on your GPU memory
        cache=True,
        project='runs/train',
        name='exp',
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
