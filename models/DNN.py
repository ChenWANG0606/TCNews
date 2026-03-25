import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from tqdm import tqdm
import numpy as np
import pandas as pd
class TorchYoutubeDNN(nn.Module):
    """
    自定义 PyTorch 双塔模型：
    - user tower: user_id embedding + 历史序列平均池化 + MLP
    - item tower: item_id embedding
    """
    def __init__(self, user_num, item_num, embedding_dim=16, hidden_units=(64, 16), padding_idx=0):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(user_num, embedding_dim)
        self.item_embedding = nn.Embedding(item_num, embedding_dim, padding_idx=padding_idx)
        self.hist_embedding = self.item_embedding  # 与 item embedding 共享参数

        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.user_mlp = nn.Sequential(*layers)

    def encode_user(self, user_id, hist_item, hist_len):
        """
        user_id: [B]
        hist_item: [B, L]
        hist_len: [B]
        """
        user_emb = self.user_embedding(user_id)  # [B, D]
        hist_emb = self.hist_embedding(hist_item)  # [B, L, D]

        mask = (hist_item != 0).float().unsqueeze(-1)  # [B, L, 1]
        # 进行平均
        hist_sum = (hist_emb * mask).sum(dim=1)        # [B, D]
        denom = hist_len.clamp(min=1).float().unsqueeze(-1)
        hist_mean = hist_sum / denom                   # [B, D]

        user_input = torch.cat([user_emb, hist_mean], dim=-1)  # [B, 2D]
        user_vec = self.user_mlp(user_input)                   # [B, D_last]
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec

    def encode_item(self, item_id):
        item_vec = self.item_embedding(item_id)
        item_vec = F.normalize(item_vec, p=2, dim=-1)
        return item_vec

    def forward(self, user_id, hist_item, hist_len, target_item):
        user_vec = self.encode_user(user_id, hist_item, hist_len)
        item_vec = self.encode_item(target_item)
        return user_vec, item_vec

class YoutubeDNNTrainer:
    def __init__(
        self,
        model,
        device,
        batch_size=256,
        lr=1e-3,
        epochs=1
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def fit(self, train_model_input):
        self.model.train()

        user_id = torch.LongTensor(train_model_input["user_id"])
        item_id = torch.LongTensor(train_model_input["click_article_id"])
        hist_id = torch.LongTensor(train_model_input["hist_article_id"])
        hist_len = torch.LongTensor(train_model_input["hist_len"])

        num_samples = len(user_id)

        for epoch in range(self.epochs):
            permutation = torch.randperm(num_samples)
            total_loss = 0.0

            for i in tqdm(range(0, num_samples, self.batch_size),
                          desc=f"Epoch {epoch+1}/{self.epochs}"):

                idx = permutation[i:i+self.batch_size]

                batch_user = user_id[idx].to(self.device)
                batch_item = item_id[idx].to(self.device)
                batch_hist = hist_id[idx].to(self.device)
                batch_hist_len = hist_len[idx].to(self.device)

                user_vec, item_vec = self.model(
                    batch_user, batch_hist, batch_hist_len, batch_item
                )

                # in-batch negative
                logits = torch.matmul(user_vec, item_vec.t()) * 20.0
                labels = torch.arange(logits.size(0), device=self.device)

                loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[Trainer] epoch {epoch+1} loss: {total_loss:.4f}")

    def get_user_embedding(self, test_model_input):
        self.model.eval()

        user_id = torch.LongTensor(test_model_input["user_id"]).to(self.device)
        hist_id = torch.LongTensor(test_model_input["hist_article_id"]).to(self.device)
        hist_len = torch.LongTensor(test_model_input["hist_len"]).to(self.device)

        user_embs = []

        with torch.no_grad():
            for i in range(0, len(user_id), 2**12):
                batch_user = user_id[i:i+2**12]
                batch_hist = hist_id[i:i+2**12]
                batch_hist_len = hist_len[i:i+2**12]

                emb = self.model.encode_user(batch_user, batch_hist, batch_hist_len)
                user_embs.append(emb.cpu().numpy())

        return np.vstack(user_embs)

    def get_item_embedding(self, item_ids):
        self.model.eval()

        item_ids = torch.LongTensor(item_ids).to(self.device)
        item_embs = []

        with torch.no_grad():
            for i in range(0, len(item_ids), 2**12):
                batch_item = item_ids[i:i+2**12]
                emb = self.model.encode_item(batch_item)
                item_embs.append(emb.cpu().numpy())

        return np.vstack(item_embs)

    def recall(self, user_embs, item_embs, topk):
        index = faiss.IndexFlatIP(item_embs.shape[1])
        index.add(item_embs.astype(np.float32))

        sim, idx = index.search(
            np.ascontiguousarray(user_embs.astype(np.float32)),
            topk
        )
        return sim, idx