import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import os

# Import the necessary components from your other scripts
# This assumes they are in the same directory or accessible in the Python path.
from dataset import LabeledStateDataset
from mlp import MLPNet, collate_batch_dense  # Re-using the model and collate function


def validate_mlp():
    """
    Loads and evaluates the trained MLPNet model checkpoints on a testing dataset.
    """
    # --- Constants ---
    # These should match the constants used during training
    ACTIONS_MAX = 128
    IS_MCTS = False

    # --- Data Loading and Vocabulary Creation ---
    # CRITICAL: We must build the feature map from the TRAINING data to ensure
    # the mapping is identical to the one used to train the model.
    training_data_path = "data/UWTempo/ver2/training.bin"
    testing_data_path = "data/UWTempo/ver2/testing.bin"

    try:
        training_ds = LabeledStateDataset(training_data_path)
        testing_ds = LabeledStateDataset(testing_data_path)
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

    # --- Create DataLoader for the TEST set ---
    # The collate function uses the feature_map from the training data.
    dl = DataLoader(testing_ds, batch_size=128, shuffle=False, num_workers=0,
                    collate_fn=lambda b: collate_batch_dense(b, feature_map))

    # --- Model Instantiation ---
    model = MLPNet(INPUT_SIZE, ACTIONS_MAX).cuda()
    model.eval()  # Set model to evaluation mode permanently for this script

    # --- Loss Functions ---
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    kld_loss = nn.KLDivLoss(reduction="batchmean")

    # --- Evaluation Loop for Each Checkpoint ---
    for epoch in range(1, 41):
        checkpoint_path = f"models/mlp_model/ckpt_{epoch}.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n--- Evaluating Checkpoint {epoch} ({checkpoint_path}) ---")
        except FileNotFoundError:
            print(f"\nSkipping epoch {epoch}: Checkpoint file not found at {checkpoint_path}.")
            continue
        except Exception as e:
            print(f"\nERROR: Could not load checkpoint for epoch {epoch}. {e}. Skipping.")
            continue

        # --- Metrics Accumulators ---
        total_p_loss, total_v_loss = 0.0, 0.0
        correct_policy_preds, total_policy_samples = 0, 0

        # --- Inference Loop ---
        with torch.no_grad():  # Essential for testing to disable gradient calculations
            for dense_input, batch_policy_labels, batch_value_labels in dl:
                # Move tensors to CUDA
                dense_input = dense_input.cuda()
                batch_policy_labels = batch_policy_labels.cuda()
                batch_value_labels = batch_value_labels.cuda()

                # Model call uses the dense input vector
                policy_logits, value_pred = model(dense_input)

                # --- Loss Calculation ---
                policy_target = batch_policy_labels
                policy_target_indices = torch.argmax(batch_policy_labels, dim=1)

                lp = ce(policy_logits, policy_target)

                lv = mse(value_pred, batch_value_labels.squeeze(-1))

                # Accumulate loss, weighted by actual batch size
                batch_size = dense_input.size(0)
                total_p_loss += lp.item() * batch_size
                total_v_loss += lv.item() * batch_size

                # --- Accuracy Calculation ---
                predicted_actions = torch.argmax(policy_logits, dim=1)
                correct_policy_preds += (predicted_actions == policy_target_indices).sum().item()
                total_policy_samples += batch_size

        # --- Print Results for the Current Checkpoint ---
        if total_policy_samples > 0:
            avg_policy_loss = total_p_loss / total_policy_samples
            avg_value_loss = total_v_loss / total_policy_samples
            accuracy = correct_policy_preds / total_policy_samples

            print(
                f"  Test Results: policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}, policy_accuracy={accuracy:.4f}")
        else:
            print("  No samples found in the testing dataset.")


if __name__ == "__main__":
    # To run this, you need:
    # 1. `dataset.py` with LabeledStateDataset.
    # 2. `mlp_net_training.py` with MLPNet and collate_batch_dense.
    # 3. Your training and testing data files in the correct paths.
    # 4. Your saved model checkpoints in `models/mlp_model/`.
    validate_mlp()
