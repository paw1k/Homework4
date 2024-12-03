from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define a simple MLP model
        input_size = n_track * 2 * 2  # Two sides (left and right), each with (x, y) for n_track points
        output_size = n_waypoints * 2  # Each waypoint has (x, y) coordinates

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, output_size)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Combine left and right boundaries
        x = torch.cat([track_left, track_right], dim=1)
        x = x.view(x.size(0), -1)

        # Normalize inputs
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        # Pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Reshape the output to (B, n_waypoints, 2)
        waypoints = x.view(x.size(0), self.n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Positional Embeddings for Tracks
        self.track_embed = nn.Linear(4, d_model)  # Adjusted to 4 input features (x, y, x, y)
        self.track_positional_encoding = nn.Parameter(torch.zeros(1, n_track, d_model))

        # Query embeddings for waypoints
        self.query_embed = nn.Parameter(torch.randn(1, n_waypoints, d_model))

        # Transformer Decoder with multiple stacked layers
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=4, dim_feedforward=128
                )
                for _ in range(3)  # Add 3 layers (depth = 3)
            ]
        )

        # Output layer for predicting waypoints
        self.output_layer = nn.Linear(d_model, 2)  # Outputs (x, y) coordinates for waypoints


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Combine track boundaries
        track_features = torch.cat([track_left, track_right], dim=-1)  # Shape: (B, n_track, 4)

        # Embed track features
        track_features = self.track_embed(track_features)  # Shape: (B, n_track, d_model)
        track_features += self.track_positional_encoding  # Add positional encoding

        # Repeat query embeddings for each batch
        query_embed = self.query_embed.repeat(track_features.size(0), 1, 1)  # Shape: (B, n_waypoints, d_model)

        # Pass through multiple transformer decoder layers
        memory = track_features.permute(1, 0, 2)  # Shape: (n_track, B, d_model)
        tgt = query_embed.permute(1, 0, 2)  # Shape: (n_waypoints, B, d_model)
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt=tgt, memory=memory)  # Shape: (n_waypoints, B, d_model)

        # Reshape back and generate waypoints
        decoder_output = tgt.permute(1, 0, 2)  # Shape: (B, n_waypoints, d_model)
        waypoints = self.output_layer(decoder_output)  # Shape: (B, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define a smaller CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Smaller filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 48, 64)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Smaller filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 24, 32)
        )

        # Fully connected layer with fewer units
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 24 * 32, 64),  # Smaller FC layer
            nn.ReLU(),
            nn.Linear(64, n_waypoints * 2)  # Predict n_waypoints (x, y) positions
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # x = image
        # x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN layers
        x = self.conv_layers(image)

        # Flatten the output from CNN layers
        x = x.view(x.size(0), -1)  # Flatten to (B, 32 * 24 * 32)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        # Reshape the output to (B, n_waypoints, 2)
        waypoints = x.view(x.size(0), self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)
    #print model size
    print(f"{model_name} model size: {model_size_mb:.2f} MB")

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
