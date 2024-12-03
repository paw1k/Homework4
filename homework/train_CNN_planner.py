import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework import load_model
from homework.datasets.road_dataset import load_data  # Ensure this exists
from homework.models import save_model
from homework.metrics import PlannerMetric  # Custom metric for trajectory planning

def train_CNN_planner(
        exp_dir: str = "logs",
        model_name: str = "cnn_planner",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 2024,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a directory for saving logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load training and validation data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Define loss function and optimizer
    loss_fn = torch.nn.L1Loss()  # Assuming regression for trajectory points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the metric for trajectory evaluation
    metrics = PlannerMetric()

    # Training loop
    for epoch in range(num_epoch):
        metrics.reset()
        model.train()
        start_time = time.time()

        for batch in train_data:
            image = batch['image'].to(device)  # Assuming 'image' is the input tensor
            waypoints = batch['waypoints'].to(device)
            labels_mask = batch['waypoints_mask'].to(device)  # Ensure the mask is available

            # Forward pass
            predictions = model(image)

            # Compute loss
            loss = loss_fn(predictions, waypoints)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            metrics.add(predictions, waypoints, labels_mask)

        # Validation
        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                image = batch['image'].to(device)
                waypoints = batch['waypoints'].to(device)
                labels_mask = batch['waypoints_mask'].to(device)

                predictions = model(image)
                metrics.add(predictions, waypoints, labels_mask)

        # Compute and log metrics
        metrics_dict = metrics.compute()

        # Print out metrics (l1_error, longitudinal_error, lateral_error) on the same line
        print(f"Epoch {epoch + 1}/{num_epoch}: "
              f"l1_error={metrics_dict.get('l1_error', 'N/A'):.4f}, "
              f"longitudinal_error={metrics_dict.get('longitudinal_error', 'N/A'):.4f}, "
              f"lateral_error={metrics_dict.get('lateral_error', 'N/A'):.4f}, ", end="")

        # Log the loss and the metrics
        logger.add_scalar("train/loss", loss.item(), epoch)

        # Log specific metrics (e.g., l1_error, longitudinal_error, lateral_error)
        for metric_name, value in metrics_dict.items():
            if metric_name != 'num_samples':  # Skip 'num_samples' as it's not a metric
                logger.add_scalar(f"val/{metric_name}", value, epoch)

        elapsed_time = time.time() - start_time
        print(f"time={elapsed_time:.2f}s")

    # Save the model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)

    # Pass all arguments to the train_CNN_planner function
    train_CNN_planner(**vars(parser.parse_args()))
