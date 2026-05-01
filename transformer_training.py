import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append("SymbolicMathematics")
from transformer_utils import CLS_Tokenizer, CLS_TransformerModel, TransformerModel, PositionalEncoding
from preprocessing_utils import build_prefix_converter, expr_to_tokens, safe_eval


# ---------------------------------------------------------------------------
# Config — edit here instead of CLI flags
# ---------------------------------------------------------------------------

CONFIG = {
    "use_cls": False,
    "data_path": "Data/cleaned_train.csv",
    "num_epochs": 1,
    "batch_size": 64,
    "lr": 4e-5,
    "data_frac": 0.1,
    "label_smoothing": 0.05,
    "entropy_bonus": 0.05,
}

SEEDS = [42]


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def preprocess_split(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df[["rules", "result"]] = df["rules"].str.split(",", n=1, expand=True)
    df["rules"] = df["rules"].str.replace("'", "")
    df["result"] = df["result"].str.replace(",", "")
    filtered = df[df["result"].isnull()].drop("result", axis=1)
    filtered.to_csv(output_path)
    df.to_csv(output_path)
    return df


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CustomTokenizer:
    def __init__(self, vocab):
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        self.token_to_id = {token: idx for idx, token in enumerate(vocab, start=1)}
        self.token_to_id[self.unk_token] = len(self.token_to_id)
        self.token_to_id[self.pad_token] = len(self.token_to_id)

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, token_list, max_length=20):
        unk_id = self.token_to_id[self.unk_token]
        pad_id = self.token_to_id[self.pad_token]

        token_ids = [self.token_to_id.get(token, unk_id) for token in token_list]
        attention_mask = [1] * len(token_ids)

        token_ids = token_ids[:max_length] + [pad_id] * (max_length - len(token_ids))
        attention_mask = attention_mask[:max_length] + [0] * (max_length - len(attention_mask))

        return token_ids, attention_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, tokenized_data, attention_masks, labels):
        self.tokenized_data = tokenized_data
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokenized_data[idx], dtype=torch.long),
            torch.tensor(self.attention_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, entropy_bonus=0.0):
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch_data, attention_mask, batch_labels in bar:
            batch_data = batch_data.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data, attention_mask)
            loss = criterion(logits, batch_labels)
            if entropy_bonus > 0.0:
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                loss = loss - entropy_bonus * entropy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == batch_labels).sum().item()
            total += batch_labels.size(0)
            bar.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_data, attention_mask, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_data, attention_mask)
                val_loss += criterion(logits, batch_labels).item()
                val_correct += (torch.argmax(logits, dim=1) == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    correct_label_counter = Counter()
    total_label_counter = Counter()

    with torch.no_grad():
        for batch_data, attention_mask, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            predicted = torch.argmax(model(batch_data, attention_mask), dim=1)
            correct_mask = predicted == batch_labels

            for true_label, is_correct in zip(batch_labels.cpu().numpy(), correct_mask.cpu().numpy()):
                total_label_counter[true_label] += 1
                if is_correct:
                    correct_label_counter[true_label] += 1

            correct += correct_mask.sum().item()
            total += batch_labels.size(0)

    acc = correct / total * 100
    print(f"\nOverall Test Accuracy: {acc:.2f}%")
    print("\nPer-label accuracy:")
    for label in sorted(total_label_counter):
        n_correct = correct_label_counter[label]
        n_total = total_label_counter[label]
        print(f"  Label {label}: {n_correct}/{n_total} ({n_correct / n_total * 100:.2f}%)")
    return acc


# ---------------------------------------------------------------------------
# Data loading (done once, shared across seeds)
# ---------------------------------------------------------------------------

def load_data(data_path: str, use_cls: bool, data_frac: float):
    data_path = Path(data_path)

    if not Path("Data/cleaned_train.csv").exists():
        preprocess_split("Data/train.csv", "Data/cleaned_train.csv")
    if not Path("Data/cleaned_test.csv").exists():
        preprocess_split("Data/test.csv", "Data/cleaned_test.csv")
    preprocess_split("Data/train.csv", "Data/cleaned_train.csv")
    preprocess_split("Data/test.csv", "Data/cleaned_test.csv")

    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} not found")

    df = pd.read_csv(data_path).sample(frac=data_frac, random_state=0)

    if "converted_function" in df.columns:
        df["converted_function"] = df["converted_function"].apply(safe_eval)
        df = df[df["converted_function"].map(len) > 0].reset_index(drop=True)
    else:
        print("No pre-computed tokens found — converting expressions to prefix notation …")
        prefix_converter = build_prefix_converter()
        token_lists, valid_indices = [], []
        for i, expr in enumerate(tqdm.tqdm(df["function"], desc="Converting")):
            try:
                token_lists.append(expr_to_tokens(prefix_converter, str(expr)))
                valid_indices.append(i)
            except Exception:
                continue
        df = df.iloc[valid_indices].reset_index(drop=True)
        df["converted_function"] = token_lists
        converted_path = data_path.with_stem(data_path.stem + "_converted")
        df.to_csv(converted_path, index=False)
        print(f"Converted tokens saved to {converted_path}")

    unique_tokens = sorted(set(t for sublist in df["converted_function"] for t in sublist))
    print(f"Vocab size: {len(unique_tokens)}")

    with open("Utils/updated_vocab.txt", "w") as f:
        f.write("\n".join(unique_tokens))

    if use_cls:
        print("Using CLS token model")
        tokenizer = CLS_Tokenizer(vocab=unique_tokens)
        vocab_size = len(unique_tokens) + 3
    else:
        print("Using mean-pooling model")
        tokenizer = CustomTokenizer(vocab=unique_tokens)
        vocab_size = len(unique_tokens) + 2

    tokenized_data, attention_masks = [], []
    for tokens in df["converted_function"]:
        ids, mask = tokenizer.encode(tokens, max_length=20)
        tokenized_data.append(ids)
        attention_masks.append(mask)

    label_encoder = LabelEncoder()
    df["rule_label"] = label_encoder.fit_transform(df["rules"])
    np.save("Utils/og_classes.npy", label_encoder.classes_)

    X = np.array(tokenized_data)
    masks = np.array(attention_masks)
    y = df["rule_label"].values

    return X, masks, y, vocab_size


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_single_seed(seed, X, masks, y, vocab_size, use_cls, num_epochs, batch_size, lr, device, tag, label_smoothing=0.25, entropy_bonus=0.05):
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_temp, m_train, m_temp, y_train, y_temp = train_test_split(
        X, masks, y, test_size=0.2, random_state=seed
    )
    X_val, X_test, m_val, m_test, y_val, y_test = train_test_split(
        X_temp, m_temp, y_temp, test_size=0.5, random_state=seed
    )

    train_loader = DataLoader(CustomDataset(X_train, m_train, y_train), batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader   = DataLoader(CustomDataset(X_val,   m_val,   y_val),   batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader  = DataLoader(CustomDataset(X_test,  m_test,  y_test),  batch_size=batch_size, shuffle=False, num_workers=1)

    if use_cls:
        model = CLS_TransformerModel(vocab_size=vocab_size).to(device)
    else:
        model = TransformerModel(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_loss, train_acc, val_loss, val_acc = train(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=num_epochs, entropy_bonus=entropy_bonus,
    )

    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size":  vocab_size,
        "use_cls":     use_cls,
    }, f"Models/{tag}_seed{seed}.pt")
    print(f"Model saved to Models/{tag}_seed{seed}.pt")

    test_acc = evaluate(model, test_loader, device)

    return {
        "seed": seed,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }


# ---------------------------------------------------------------------------
# Multi-seed plot
# ---------------------------------------------------------------------------

def plot_results(all_results, num_epochs):
    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors

    for i, r in enumerate(all_results):
        c = colors[i % len(colors)]
        label = f"seed {r['seed']}"
        ax1.plot(epochs, r["train_loss"], color=c, alpha=0.4, linestyle="--")
        ax1.plot(epochs, r["val_loss"],   color=c, alpha=0.9, label=label)
        ax2.plot(epochs, r["train_acc"],  color=c, alpha=0.4, linestyle="--")
        ax2.plot(epochs, r["val_acc"],    color=c, alpha=0.9, label=label)

    # Mean across seeds
    mean_train_loss = np.mean([r["train_loss"] for r in all_results], axis=0)
    mean_val_loss   = np.mean([r["val_loss"]   for r in all_results], axis=0)
    mean_train_acc  = np.mean([r["train_acc"]  for r in all_results], axis=0)
    mean_val_acc    = np.mean([r["val_acc"]    for r in all_results], axis=0)

    ax1.plot(epochs, mean_train_loss, color="black", linewidth=2, linestyle="--", label="mean train")
    ax1.plot(epochs, mean_val_loss,   color="black", linewidth=2, label="mean val")
    ax2.plot(epochs, mean_train_acc,  color="black", linewidth=2, linestyle="--", label="mean train")
    ax2.plot(epochs, mean_val_acc,    color="black", linewidth=2, label="mean val")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss");         ax1.set_title("Loss Over Epochs");     ax1.legend(); ax1.grid()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.set_title("Accuracy Over Epochs"); ax2.legend(); ax2.grid()

    plt.tight_layout()
    plt.savefig("Models/training_metrics.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_experiments(seeds=SEEDS, **config):
    use_cls        = config.get("use_cls",        CONFIG["use_cls"])
    data_path      = config.get("data_path",      CONFIG["data_path"])
    num_epochs     = config.get("num_epochs",     CONFIG["num_epochs"])
    batch_size     = config.get("batch_size",     CONFIG["batch_size"])
    lr             = config.get("lr",             CONFIG["lr"])
    data_frac      = config.get("data_frac",      CONFIG["data_frac"])
    label_smoothing = config.get("label_smoothing", CONFIG["label_smoothing"])
    entropy_bonus  = config.get("entropy_bonus",  CONFIG["entropy_bonus"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, masks, y, vocab_size = load_data(data_path, use_cls, data_frac)
    tag = f"{'cls' if use_cls else 'full'}_{Path(data_path).stem}"

    all_results = []
    for seed in seeds:
        result = run_single_seed(
            seed, X, masks, y, vocab_size,
            use_cls=use_cls, num_epochs=num_epochs,
            batch_size=batch_size, lr=lr,
            device=device, tag=tag,
            label_smoothing=label_smoothing,
            entropy_bonus=entropy_bonus,
        )
        all_results.append(result)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for r in all_results:
        print(f"  Seed {r['seed']}: test acc = {r['test_acc']:.2f}%")
    mean_test = np.mean([r["test_acc"] for r in all_results])
    print(f"  Mean test acc: {mean_test:.2f}%")

    plot_results(all_results, num_epochs)


if __name__ == "__main__":
    run_experiments(seeds=SEEDS)
