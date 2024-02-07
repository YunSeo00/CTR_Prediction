import time
import os
import sys
import argparse

import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model.logistic_regression import LogisticRegressionWithEmbedding
from model.ffm import FieldAwareFactorizationMachineModel
from model.deepfm import DeepFactorizationMachineModel
from model.deepffm import DeepFieldAwareFactorizationMachineModel
from model.tab_transformer import TabTransformer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def readArguments(opts=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ffm")
    parser.add_argument("--verbose_interval", type=int, default=1)
    parser.add_argument("--data", type=str, default="ip_feature_add_domain")

    # common arguments
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default= 8)

    # deepfm arguments
    parser.add_argument("--mlp_dims", nargs="?", default="[32, 32]")
    parser.add_argument("--dropout", type=float, default=0.5)

    # tabtransformer arguments
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dropout", type=float, default=0.3)
    parser.add_argument("--attn_dropout", type=float, default=0.3)
    
    return parser.parse_args(opts)


args = readArguments(sys.argv[1:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"devide: {device}")


## Save numpy vector
X_train = np.load(f"./data/{args.data}/X_train.npy")
y_train = np.load(f"./data/{args.data}/y_train.npy")
X_val = np.load(f"./data/{args.data}/X_val.npy")
y_val = np.load(f"./data/{args.data}/y_val.npy")
X_test = np.load(f"./data/{args.data}/X_test.npy")
offsets = np.load(f"./data/{args.data}/offsets.npy")
total_field_dims = np.load(f"./data/{args.data}/total_field_dims.npy")


X_train = torch.IntTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.IntTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.IntTensor(X_test)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test)

# train model
offsets = torch.IntTensor(offsets).to(device)

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs

print(f"offsets: {offsets}, total_field_dims: {total_field_dims}, embed_dim: {args.embed_dim}, batch_size: {batch_size}, learning_rate: {learning_rate}, epochs: {epochs}")

if args.model == "logistic":
    model = LogisticRegressionWithEmbedding(offsets, total_field_dims, 1, args.embed_dim).to(device)
    filename = f"logistic_embedding_{args.embed_dim}_lr_{args.learning_rate}_batch_{args.batch_size}_data_{args.data}"
    fname = f"./result/{filename}"
    if not os.path.isdir(fname):
        os.makedirs(fname, exist_ok=False)

if args.model == "ffm":
    model = FieldAwareFactorizationMachineModel(offsets, total_field_dims, args.embed_dim).to(device)
    filename = f"ffm_embedding_{args.embed_dim}_lr_{args.learning_rate}_batch_{args.batch_size}_data_{args.data}"
    fname = f"./result/{filename}"
    if not os.path.isdir(fname):
        os.makedirs(fname, exist_ok=False)

if args.model == "deepfm":
    mlp_dims = eval(args.mlp_dims)
    model = DeepFactorizationMachineModel(offsets, total_field_dims, args.embed_dim, mlp_dims, args.dropout).to(device)
    filename = f'deepfm_embedding_{args.embed_dim}_lr_{args.learning_rate}_batch_{args.batch_size}_mlp_{"-".join([str(i) for i in mlp_dims])}_dropout_{args.dropout}_data_{args.data}'
    fname = f"./result/{filename}"
    if not os.path.isdir(fname):
        os.makedirs(fname, exist_ok=False)
        
if args.model == "deepffm":
    mlp_dims = eval(args.mlp_dims)
    model = DeepFieldAwareFactorizationMachineModel(offsets, total_field_dims, args.embed_dim, mlp_dims, args.dropout).to(device)
    filename = f'deepffm_embedding_{args.embed_dim}_lr_{args.learning_rate}_batch_{args.batch_size}_mlp_{"-".join([str(i) for i in mlp_dims])}_dropout_{args.dropout}_data_{args.data}'
    fname = f"./result/{filename}"
    if not os.path.isdir(fname):
        os.makedirs(fname, exist_ok=False)

if args.model == "tabtransformer":
    field_dims = np.load(f"./data/{args.data}/field_dims.npy")
    model = TabTransformer(offsets, field_dims, args.embed_dim, args.depth, args.heads, use_shared_categ_embed=True, ff_dropout=args.ff_dropout, attn_dropout=args.attn_dropout).to(device)
    filename = f'tabtransformer_lr_{args.learning_rate}_batch_{args.batch_size}_data_{args.data}_dim_{args.embed_dim}_depth_{args.depth}_heads_{args.heads}_ff_dropout_{args.ff_dropout}_attn_dropout_{args.attn_dropout}'
    fname = f"./result/{filename}"
    if not os.path.isdir(fname):
        os.makedirs(fname, exist_ok=False)

# count model params
print(f"Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

early_stopping_th = 0
max_val_auc = 0
for epoch in range(epochs + 1):
    start_time = time.time()
    model.train()

    loss_list = []
    hypothesis_list = []
    y_train_list = []

    for X_train_batch, y_train_batch in tqdm.tqdm(train_loader):
    # for X_train_batch, y_train_batch in train_loader:
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)
        hypothesis = model(X_train_batch)
        loss = criterion(hypothesis, y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().item())
        hypothesis_list.append(hypothesis.detach().cpu())
        y_train_list.append(y_train_batch.detach().cpu())
        # print(roc_auc_score(y_train_batch.detach().cpu(), hypothesis.detach().cpu()))
    train_auc = roc_auc_score(torch.cat(y_train_list), torch.cat(hypothesis_list))

    with open(fname + "/log.txt", "a") as f:
        f.write(f"1 Epoch time: {time.time() - start_time:.2f} sec\n")

    if epoch % args.verbose_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss_list = []
            val_hypothesis_list = []
            y_val_list = []

            #for X_val_batch, y_val_batch in tqdm.tqdm(val_loader):
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                val_hypothesis = model(X_val_batch)
                val_loss = criterion(val_hypothesis, y_val_batch)

                val_loss_list.append(val_loss.item())
                val_hypothesis_list.append(val_hypothesis.cpu())
                y_val_list.append(y_val_batch.cpu())

            val_auc = roc_auc_score(torch.cat(y_val_list), torch.cat(val_hypothesis_list))

            train_loss_result = np.mean(loss_list)
            val_loss_result = np.mean(val_loss_list)

            with open(fname + "/log.txt", "a") as f:
                f.write(
                    f"Epoch {epoch}/{epochs} - Loss: {train_loss_result:.6f}, Validation Loss: {val_loss_result:.6f}, AUC: {train_auc:.6f}, Validation AUC: {val_auc:.6f}\n"
                )

            if val_auc > max_val_auc:
                with open(fname + "/log.txt", "a") as f:
                    f.write("Model Saved\n")
                torch.save(model.state_dict(), fname + "/model_weights.pth")
                model.eval()
                with torch.no_grad():
                    test_hypothesis_list = []

                    # for X_test_batch in tqdm.tqdm(test_loader):
                    for X_test_batch in test_loader:
                        X_test_batch = X_test_batch[0].to(device)
                        test_hypothesis = model(X_test_batch)
                        test_hypothesis_list.append(test_hypothesis.cpu())

                submission = pd.read_csv("../input/sample_submission.csv")
                submission["click"] = torch.concat(test_hypothesis_list).numpy()
                submission.to_csv(fname + f"/{filename}_submission.csv", index=False)

                early_stopping_th = 0
            max_val_auc = max(max_val_auc, val_auc)
        early_stopping_th += 1
        if early_stopping_th >= args.early_stopping:
            with open(fname + "/log.txt", "a") as f:
                f.write("Early Stopping\n")
            break
