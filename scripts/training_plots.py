import os
import json
import matplotlib.pyplot as plt

# ── Dirs ──────────────────────────────────────────────────────────────────────
base_dir = "/home/henrich/Documents/sofiax"
fig_dir  = os.path.join(base_dir, "figs")
os.makedirs(fig_dir, exist_ok=True)

# Find the latest checkpoint in your results folder
results_dir = "./results"
checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
checkpoints.sort(key=lambda x: int(x.split("-")[1]))
latest_checkpoint = checkpoints[-1]

state_path = os.path.join(results_dir, latest_checkpoint, "trainer_state.json")

print(f"Reading logs from {state_path}...")

with open(state_path, "r") as f:
    state = json.load(f)

train_steps, train_losses, lrs = [], [], []
val_steps, val_losses = [], []

# Parse the log history
for log in state["log_history"]:
    if "loss" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])
        lrs.append(log.get("learning_rate", 0))
    elif "eval_loss" in log:
        val_steps.append(log["step"])
        val_losses.append(log["eval_loss"])

# ── Plot 1: Convergence ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, label="Train Loss", linewidth=1.5)
if val_losses:
    plt.plot(val_steps, val_losses, label="Validation Loss", color="orange", 
             linewidth=2, marker="o", markersize=4)
plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.title("Training Convergence — ChemDFM-v1.5-8B Odor Fine-tune")
plt.legend()
plt.grid(True)
plt.tight_layout()
conv_path = os.path.join(fig_dir, "convergence.png")
plt.savefig(conv_path, dpi=150)
print(f"Saved: {conv_path}")

# ── Plot 2: Learning rate schedule ───────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(train_steps, lrs, color="green", linewidth=1.5)
plt.xlabel("Global Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule (Cosine)")
plt.grid(True)
plt.tight_layout()
lr_path = os.path.join(fig_dir, "learning_rate.png")
plt.savefig(lr_path, dpi=150)
print(f"Saved: {lr_path}")

print("Recovery complete!")