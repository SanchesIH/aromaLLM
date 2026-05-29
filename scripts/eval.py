import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# ── Dirs ──────────────────────────────────────────────────────────────────────
base_dir = "/home/henrich/Documents/sofiax"
fig_dir  = os.path.join(base_dir, "figs")
metrics_dir = os.path.join(base_dir, "metrics")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# ── Find the latest trainer state ─────────────────────────────────────────────
results_dir = "./results"
checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
checkpoints.sort(key=lambda x: int(x.split("-")[1]))
latest_checkpoint = checkpoints[-1]
state_path = os.path.join(results_dir, latest_checkpoint, "trainer_state.json")

print(f"Reading logs from {state_path}...")
with open(state_path, "r") as f:
    state = json.load(f)

# ── Data extraction arrays ────────────────────────────────────────────────────
train_steps, train_losses, train_acc, train_entropy = [], [], [], []
val_steps, val_losses, val_acc, val_entropy = [], [], [], []

for log in state["log_history"]:
    if "loss" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])
        train_acc.append(log.get("mean_token_accuracy", None))
        train_entropy.append(log.get("entropy", None))
    elif "eval_loss" in log:
        val_steps.append(log["step"])
        val_losses.append(log["eval_loss"])
        val_acc.append(log.get("eval_mean_token_accuracy", None))
        val_entropy.append(log.get("eval_entropy", None))

# ── SAVE ALL METRICS TO CSV ───────────────────────────────────────────────────
train_df = pd.DataFrame({
    "step": train_steps,
    "train_loss": train_losses,
    "train_accuracy": train_acc,
    "train_entropy": train_entropy
})

val_df = pd.DataFrame({
    "step": val_steps,
    "val_loss": val_losses,
    "val_accuracy": val_acc,
    "val_entropy": val_entropy
})

full_metrics_df = pd.merge(train_df, val_df, on="step", how="outer").sort_values("step")
csv_path = os.path.join(metrics_dir, "full_training_metrics.csv")
full_metrics_df.to_csv(csv_path, index=False)
print(f"✓ Saved full detailed metrics CSV to: {csv_path}")

# Handle potential missing test metrics
test_loss = None
test_csv_path = os.path.join(metrics_dir, "test_metrics.csv")
if os.path.exists(test_csv_path):
    test_data = pd.read_csv(test_csv_path)
    if not test_data.empty and 'test_loss' in test_data.columns:
        test_loss = test_data['test_loss'].iloc[0]

# ── Create a 2x2 Plot Dashboard ───────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("ChemDFM Odor Fine-Tuning: Full Performance Dashboard", fontsize=16, fontweight='bold')

# Plot 1: Loss Convergence (Top Left)
axs[0, 0].plot(train_steps, train_losses, label="Train Loss", linewidth=1.5, color="tab:blue")
if val_losses:
    axs[0, 0].plot(val_steps, val_losses, label="Val Loss", marker="o", color="tab:orange", markersize=4)
if test_loss:
    axs[0, 0].axhline(y=test_loss, color="tab:red", linestyle="--", label=f"Final Test Loss: {test_loss:.4f}")
axs[0, 0].set_title("Cross-Entropy Loss (Lower is better)")
axs[0, 0].set_xlabel("Global Step")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot 2: Token Accuracy (Top Right)
if train_acc and train_acc[0] is not None:
    axs[0, 1].plot(train_steps, [float(x) for x in train_acc], label="Train Accuracy", color="tab:blue")

if val_acc and val_acc[0] is not None:
    float_val_acc = [float(x) for x in val_acc]
    max_acc = max(float_val_acc)
    # The peak accuracy is now appended directly to the legend label here
    axs[0, 1].plot(val_steps, float_val_acc, label=f"Val Accuracy (Peak: {max_acc:.4f})", marker="o", color="tab:orange", markersize=4)

axs[0, 1].set_title("Next-Token Accuracy (Higher is better)")
axs[0, 1].set_xlabel("Global Step")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].grid(True)
axs[0, 1].legend(loc="lower right")

# Plot 3: Entropy (Bottom Left)
if train_entropy and train_entropy[0] is not None:
    axs[1, 0].plot(train_steps, [float(x) for x in train_entropy], label="Train Entropy", color="tab:purple")
if val_entropy and val_entropy[0] is not None:
    axs[1, 0].plot(val_steps, [float(x) for x in val_entropy], label="Val Entropy", marker="o", color="tab:pink", markersize=4)
axs[1, 0].set_title("Prediction Entropy / Uncertainty (Lower is better)")
axs[1, 0].set_xlabel("Global Step")
axs[1, 0].set_ylabel("Entropy")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot 4: Final Summary Bar Chart (Bottom Right)
labels = ['Final Train Loss', 'Final Val Loss']
values = [train_losses[-1], val_losses[-1]] if val_losses else [train_losses[-1], 0]
colors = ['tab:blue', 'tab:orange']

if test_loss:
    labels.append('Test Loss')
    values.append(test_loss)
    colors.append('tab:red')

axs[1, 1].bar(labels, values, color=colors)
axs[1, 1].set_title("Final Loss Comparison Across Splits")
axs[1, 1].set_ylabel("Loss")

# Add 15% headroom to the y-axis so labels don't get cut off
max_val = max(values)
axs[1, 1].set_ylim(0, max_val * 1.15) 

for i, v in enumerate(values):
    axs[1, 1].text(i, v + (max_val * 0.02), f"{v:.4f}", ha='center', fontweight='bold')

# Format and Save
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
dashboard_path = os.path.join(fig_dir, "full_performance_dashboard.png")
plt.savefig(dashboard_path, dpi=200)
print(f"✓ Generated comprehensive dashboard: {dashboard_path}")