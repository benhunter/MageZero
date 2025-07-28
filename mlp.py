import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import os
import struct

# Import your actual LabeledStateDataset and original collate_batch
# This assumes the code you provided is in a file named `dataset.py`
from dataset import LabeledStateDataset


# ==============================================================================
# New MLP Model
# ==============================================================================

class MLPNet(nn.Module):
    """
    A simple MLP model that takes a dense multi-hot encoded vector as input.
    """

    def __init__(self, input_size, policy_size_A):
        super().__init__()

        hidden_dim1 = 512
        hidden_dim2 = 256

        self.mlp_body = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim2, policy_size_A)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x is the dense input tensor of shape [batch_size, input_size]
        h = self.mlp_body(x)
        policy_logits = self.policy_head(h)
        value_pred = self.value_head(h).squeeze(-1)
        return policy_logits, value_pred


# ==============================================================================
# New Collate Function for MLP
# ==============================================================================

def collate_batch_dense(batch, feature_map):
    """
    Custom collate function to convert sparse indices into a dense multi-hot vector.
    This function uses a pre-computed feature_map to map original indices to a new
    dense range [0, vocab_size-1].
    """
    vocab_size = len(feature_map)
    dense_vectors = []
    policy_list = []
    value_list = []

    # Your LabeledStateDataset's __getitem__ returns a tuple.
    # We unpack that tuple here for each sample in the batch.
    for _indices, _policy, _value in batch:
        # Create a zero vector for the current sample, sized to the new vocabulary
        dense_vec = torch.zeros(vocab_size)

        # Check if there are any indices to process.
        if _indices.numel() > 0:
            # For each original index, find its new mapped index and set it to 1.0
            for original_idx in _indices:
                mapped_idx = feature_map.get(original_idx.item())
                # Only set the value if the feature was in the training vocabulary
                if mapped_idx is not None:
                    dense_vec[mapped_idx] = 1.0

        dense_vectors.append(dense_vec)
        policy_list.append(_policy)
        value_list.append(_value)

    # Stack the lists of tensors into batches
    dense_batch = torch.stack(dense_vectors)
    policies = torch.stack(policy_list)
    values = torch.stack(value_list)

    dense_batch = dense_batch.mul(2.0).sub(1.0)

    return dense_batch, policies, values


# ==============================================================================
# New Training Function for MLP
# ==============================================================================

def train_mlp():
    # --- Constants ---
    ACTIONS_MAX = 128  # Should match the 'A' from your dataset header
    IS_MCTS = False

    # --- Data Loading and Vocabulary Creation ---
    # Make sure the path to your data is correct.
    try:
        training_ds = LabeledStateDataset("data/UWTempo/ver1/training.bin")
        testing_ds = LabeledStateDataset("data/UWTempo/ver1/testing.bin")
    except (FileNotFoundError, IOError) as e:
        print(f"FATAL: Could not load dataset. {e}")
        return

    print("Scanning TRAINING and TESTING datasets to build a combined feature vocabulary...")
    all_features = set()

    # Add all features from the training set
    for indices, _, _ in training_ds:
        all_features.update(indices.tolist())

    # Add all features from the testing set
    for indices, _, _ in testing_ds:
        all_features.update(indices.tolist())

    sorted_features = sorted(list(all_features))
    feature_map = {original_idx: new_idx for new_idx, original_idx in enumerate(sorted_features)}

    INPUT_SIZE = len(feature_map)
    print(f"Combined vocabulary built. Found {INPUT_SIZE} unique features across both datasets.")

    # Step 3: Create the DataLoader, passing the feature_map to the collate function.
    dl = DataLoader(training_ds, batch_size=128, shuffle=True, num_workers=0,
                    collate_fn=lambda b: collate_batch_dense(b, feature_map))

    # --- Model and Optimizer ---
    model = MLPNet(INPUT_SIZE, ACTIONS_MAX).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Checkpoint Loading ---
    checkpoint_path = "models/mlp_model/ckpt_0.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint. {e}. Starting from scratch.")

    # --- Loss Functions ---
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    kld_loss = nn.KLDivLoss(reduction="batchmean")

    # --- Training Loop ---
    for epoch in range(1, 41):
        total_p_loss, total_v_loss = 0.0, 0.0
        model.train()

        for dense_input, batch_policy_labels, batch_value_labels in dl:
            dense_input = dense_input.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()

            policy_logits, value_pred = model(dense_input)


            policy_target = batch_policy_labels
            lp = ce(policy_logits, policy_target)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            loss = lp+lv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_p_loss += lp.item()
            total_v_loss += lv.item()

        avg_p_loss = total_p_loss / len(dl)
        avg_v_loss = total_v_loss / len(dl)
        print(f"Epoch {epoch}  policy_loss={avg_p_loss:.3f}  value_loss={avg_v_loss:.3f}")

        # --- Checkpoint Saving ---
        os.makedirs("models/mlp_model", exist_ok=True)
        checkpoint_save_path = f"models/mlp_model/ckpt_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_p_loss': avg_p_loss,
            'avg_v_loss': avg_v_loss,
        }, checkpoint_save_path)


if __name__ == "__main__":
    # To run this, you would need your actual LabeledStateDataset class
    # in dataset.py and the corresponding data files.
    train_mlp()

