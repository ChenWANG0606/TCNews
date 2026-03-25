import argparse
import collections
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from metrics import metrics_recall
from models.CF import item_based_recommend, itemcf_sim, user_based_recommend
from models.DNN import TorchYoutubeDNN, YoutubeDNNTrainer, u2u_embdding_sim
from models.VectorSim import embdding_sim
from utils.data_utils import (
    bucketize_words_count,
    gen_data_set,
    gen_model_input,
    get_all_click_sample,
    get_all_click_df,
    get_hist_and_last_click,
    get_item_emb_dict,
    get_item_info_df,
    get_item_info_dict,
    get_item_topk_click,
    get_user_hist_item_info_dict,
    get_user_item_time,
)


RECALL_METHODS = (
    "itemcf_sim_itemcf_recall",
    "embedding_sim_item_recall",
    "youtubednn_recall",
    "youtubednn_usercf_recall",
    "cold_start_recall",
)


def log_step(message):
    print(f"\n[Pipeline] {message}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_series(series):
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
    return ((series - min_value) / (max_value - min_value)).astype(np.float32)


def prepare_click_df(data_path, offline, use_sample=False, sample_nums=10000):
    if use_sample:
        log_step(f"Loading sampled click data via get_all_click_sample(sample_nums={sample_nums})")
    else:
        log_step(f"Loading full click data via get_all_click_df(offline={offline})")
    if use_sample:
        all_click_df = get_all_click_sample(data_path=data_path, sample_nums=sample_nums).copy()
    else:
        all_click_df = get_all_click_df(data_path=data_path, offline=offline).copy()
    all_click_df = all_click_df.sort_values(["user_id", "click_timestamp"]).reset_index(drop=True)
    all_click_df["click_timestamp"] = normalize_series(all_click_df["click_timestamp"])
    log_step(
        "Click data ready: "
        f"rows={len(all_click_df)}, "
        f"users={all_click_df['user_id'].nunique()}, "
        f"items={all_click_df['click_article_id'].nunique()}"
    )
    return all_click_df


def load_artifacts(data_path, save_path, offline, use_sample=False, sample_nums=10000):
    ensure_dir(save_path)
    log_step(f"Preparing artifacts from data_path={data_path}, save_path={save_path}")
    all_click_df = prepare_click_df(
        data_path=data_path,
        offline=offline,
        use_sample=use_sample,
        sample_nums=sample_nums,
    )
    item_info_df = get_item_info_df(data_path).copy()
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df.copy())
    item_emb_dict = get_item_emb_dict(data_path, save_path)
    item_emb_df = pd.read_csv(os.path.join(data_path, "articles_emb.csv"))
    log_step(
        "Item artifacts ready: "
        f"item_info={len(item_info_df)}, "
        f"item_emb={len(item_emb_df)}"
    )
    return {
        "all_click_df": all_click_df,
        "item_info_df": item_info_df,
        "item_type_dict": item_type_dict,
        "item_words_dict": item_words_dict,
        "item_created_time_dict": item_created_time_dict,
        "item_emb_dict": item_emb_dict,
        "item_emb_df": item_emb_df,
    }


def get_eval_frames(all_click_df, metric_recall):
    if metric_recall:
        log_step("Metric recall enabled, splitting history clicks and last clicks for offline evaluation")
        hist_df, last_df = get_hist_and_last_click(all_click_df)
        log_step(f"Evaluation split ready: hist_rows={len(hist_df)}, last_click_rows={len(last_df)}")
        return hist_df, last_df
    log_step("Metric recall disabled, using all click data for recall")
    return all_click_df, None


def save_pickle(obj, save_path, filename):
    ensure_dir(save_path)
    with open(os.path.join(save_path, filename), "wb") as file_obj:
        pickle.dump(obj, file_obj)


def run_itemcf_recall(click_df, item_created_time_dict, emb_i2i_sim, save_path, sim_item_topk=20, recall_item_num=10):
    log_step(f"Running ItemCF recall: sim_item_topk={sim_item_topk}, recall_item_num={recall_item_num}")
    i2i_sim = itemcf_sim(click_df, item_created_time_dict, save_path)
    user_item_time_dict = get_user_item_time(click_df)
    item_topk_click = get_item_topk_click(click_df, k=50)
    user_recall_items_dict = {}
    for user in tqdm(click_df["user_id"].unique(), desc="itemcf_recall"):
        user_recall_items_dict[user] = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim,
        )
    save_pickle(user_recall_items_dict, save_path, "itemcf_recall_dict.pkl")
    log_step(f"ItemCF recall finished for {len(user_recall_items_dict)} users")
    return user_recall_items_dict


def run_embedding_recall(click_df, item_created_time_dict, item_emb_df, save_path, sim_item_topk=150, recall_item_num=100, topk=10):
    log_step(
        "Running embedding i2i recall: "
        f"faiss_topk={topk}, sim_item_topk={sim_item_topk}, recall_item_num={recall_item_num}"
    )
    emb_i2i_sim = embdding_sim(click_df, item_emb_df, save_path, topk=topk)
    user_item_time_dict = get_user_item_time(click_df)
    item_topk_click = get_item_topk_click(click_df, k=50)
    user_recall_items_dict = {}
    for user in tqdm(click_df["user_id"].unique(), desc="embedding_recall"):
        user_recall_items_dict[user] = item_based_recommend(
            user,
            user_item_time_dict,
            emb_i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim,
        )
    save_pickle(user_recall_items_dict, save_path, "embedding_sim_item_recall.pkl")
    log_step(f"Embedding i2i recall finished for {len(user_recall_items_dict)} users")
    return emb_i2i_sim, user_recall_items_dict


def youtubednn_u2i_dict(data, item_info_df, save_path, topk=20, seq_len=30, words_bucket_num=20, epochs=10):
    log_step(
        "Running YoutubeDNN u2i recall: "
        f"topk={topk}, seq_len={seq_len}, words_bucket_num={words_bucket_num}, epochs={epochs}"
    )
    data = data.merge(
        item_info_df[["click_article_id", "category_id", "words_count"]],
        how="left",
        on="click_article_id",
    ).copy()
    data["words_count"] = bucketize_words_count(data["words_count"], bucket_num=words_bucket_num)

    raw_user_profile = data[["user_id"]].drop_duplicates("user_id").copy()
    raw_item_profile = data[["click_article_id"]].drop_duplicates("click_article_id").copy()

    features = ["click_article_id", "user_id", "category_id", "words_count"]
    feature_max_idx = {}
    encoders = {}
    for feature in features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = int(data[feature].max()) + 1
        encoders[feature] = encoder

    user_profile = data[["user_id"]].drop_duplicates("user_id").copy()
    item_profile = data[["click_article_id", "category_id", "words_count"]].drop_duplicates("click_article_id").copy()

    user_index_2_rawid = dict(zip(user_profile["user_id"], raw_user_profile["user_id"]))
    item_index_2_rawid = dict(zip(item_profile["click_article_id"], raw_item_profile["click_article_id"]))
    user_raw_hist = (
        data.groupby("user_id")["click_article_id"]
        .apply(list)
        .reset_index(name="hist_items")
    )
    user_raw_hist["user_id"] = user_raw_hist["user_id"].map(user_index_2_rawid)
    user_hist_item_dict = {
        row["user_id"]: {item_index_2_rawid[item] for item in row["hist_items"]}
        for _, row in user_raw_hist.iterrows()
    }

    item_feat_dict = {
        "category_id": dict(zip(item_profile["click_article_id"], item_profile["category_id"])),
        "words_count": dict(zip(item_profile["click_article_id"], item_profile["words_count"])),
    }

    train_set, test_set = gen_data_set(data, 0)
    log_step(f"YoutubeDNN samples ready: train={len(train_set)}, test={len(test_set)}")
    train_model_input, _ = gen_model_input(train_set, user_profile, seq_len, item_feat_dict=item_feat_dict)
    test_model_input, _ = gen_model_input(test_set, user_profile, seq_len, item_feat_dict=item_feat_dict)

    embedding_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchYoutubeDNN(
        user_num=feature_max_idx["user_id"],
        item_num=feature_max_idx["click_article_id"],
        embedding_dim=embedding_dim,
        hidden_units=(128, embedding_dim),
        item_hidden_units=(128, embedding_dim),
        padding_idx=0,
        category_num=feature_max_idx["category_id"],
        words_num=feature_max_idx["words_count"],
    ).to(device)

    trainer = YoutubeDNNTrainer(model, device, batch_size=512, lr=1e-3, epochs=epochs)
    log_step(f"YoutubeDNN training started on device={device}")
    trainer.fit(train_model_input)

    log_step("Extracting YoutubeDNN embeddings")
    user_embs = trainer.get_user_embedding(test_model_input)
    all_item_ids = item_profile["click_article_id"].values
    all_item_categories = item_profile["category_id"].values
    all_item_words = item_profile["words_count"].values
    item_embs = trainer.get_item_embedding(
        all_item_ids,
        item_category_ids=all_item_categories,
        item_words_ids=all_item_words,
    )

    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    raw_user_id_emb_dict = {user_index_2_rawid[k]: v for k, v in zip(test_model_input["user_id"], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: v for k, v in zip(item_profile["click_article_id"], item_embs)}

    save_pickle(raw_user_id_emb_dict, save_path, "user_youtube_emb.pkl")
    save_pickle(raw_item_id_emb_dict, save_path, "item_youtube_emb.pkl")

    log_step("Running FAISS retrieval for YoutubeDNN u2i")
    search_topk = min(max(topk * 3, topk + 20), len(item_embs))
    sim, idx = trainer.recall(user_embs, item_embs, search_topk)

    user_recall_items_dict = collections.defaultdict(list)
    item_profile_ids = item_profile["click_article_id"].tolist()
    for target_idx, sim_value_list, rele_idx_list in tqdm(
        zip(test_model_input["user_id"], sim, idx),
        total=len(test_model_input["user_id"]),
        desc="youtube_u2i_recall",
    ):
        target_raw_id = user_index_2_rawid[target_idx]
        hist_items = user_hist_item_dict.get(target_raw_id, set())
        ranked_items = []
        for rele_idx, sim_value in zip(rele_idx_list, sim_value_list):
            if rele_idx == -1:
                continue
            rele_raw_id = item_index_2_rawid[item_profile_ids[rele_idx]]
            if rele_raw_id in hist_items:
                continue
            ranked_items.append((rele_raw_id, float(sim_value)))
            if len(ranked_items) == topk:
                break
        user_recall_items_dict[target_raw_id] = ranked_items

    user_recall_items_dict = dict(user_recall_items_dict)
    save_pickle(user_recall_items_dict, save_path, "youtube_u2i_dict.pkl")
    log_step(f"YoutubeDNN u2i recall finished for {len(user_recall_items_dict)} users")
    return user_recall_items_dict, raw_user_id_emb_dict


def run_youtube_usercf_recall(click_df, user_emb_dict, item_created_time_dict, emb_i2i_sim, save_path, sim_user_topk=20, recall_item_num=10, topk=10):
    log_step(
        "Running YoutubeDNN usercf recall: "
        f"faiss_topk={topk}, sim_user_topk={sim_user_topk}, recall_item_num={recall_item_num}"
    )
    u2u_sim = u2u_embdding_sim(click_df, user_emb_dict, save_path, topk=topk)
    user_item_time_dict = get_user_item_time(click_df)
    item_topk_click = get_item_topk_click(click_df, k=50)
    user_recall_items_dict = {}
    for user in tqdm(click_df["user_id"].unique(), desc="youtube_usercf_recall"):
        if user not in u2u_sim or user not in user_item_time_dict:
            continue
        user_recall_items_dict[user] = user_based_recommend(
            user,
            user_item_time_dict,
            u2u_sim,
            sim_user_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim,
        )
    save_pickle(user_recall_items_dict, save_path, "youtubednn_usercf_recall.pkl")
    log_step(f"YoutubeDNN usercf recall finished for {len(user_recall_items_dict)} users")
    return user_recall_items_dict


def cold_start_items(
    user_recall_items_dict,
    user_hist_item_typs_dict,
    user_hist_item_ids_dict,
    user_hist_item_words_dict,
    user_last_item_created_time_dict,
    item_type_dict,
    item_words_dict,
    item_created_time_dict,
    click_article_ids_set,
    recall_item_num,
    words_window=200,
    created_time_window=0.25,
):
    log_step(
        "Running cold start filter: "
        f"candidate_users={len(user_recall_items_dict)}, recall_item_num={recall_item_num}"
    )
    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items(), desc="cold_start_filter"):
        hist_item_type_set = user_hist_item_typs_dict.get(user, set())
        hist_item_ids = user_hist_item_ids_dict.get(user, set())
        hist_mean_words = user_hist_item_words_dict.get(user)
        hist_last_item_created_time = user_last_item_created_time_dict.get(user)
        if hist_mean_words is None or hist_last_item_created_time is None:
            cold_start_user_items_dict[user] = []
            continue

        filtered_items = []
        for item, score in item_list:
            curr_item_type = item_type_dict.get(item)
            curr_item_words = item_words_dict.get(item)
            curr_item_created_time = item_created_time_dict.get(item)
            if curr_item_type is None or curr_item_words is None or curr_item_created_time is None:
                continue
            if curr_item_type not in hist_item_type_set:
                continue
            if item in hist_item_ids or item in click_article_ids_set:
                continue
            if abs(curr_item_words - hist_mean_words) > words_window:
                continue
            if abs(curr_item_created_time - hist_last_item_created_time) > created_time_window:
                continue
            filtered_items.append((item, score))

        cold_start_user_items_dict[user] = sorted(filtered_items, key=lambda x: x[1], reverse=True)[:recall_item_num]

    log_step(f"Cold start filter finished for {len(cold_start_user_items_dict)} users")
    return cold_start_user_items_dict


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25, save_path=None):
    log_step(f"Combining multi-recall results with final_topk={topk}")
    final_recall_items_dict = {}

    def normalize_scores(sorted_item_list):
        if len(sorted_item_list) < 2:
            return sorted_item_list
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        if max_sim <= min_sim:
            return [(item, 1.0) for item, _ in sorted_item_list]
        return [(item, (score - min_sim) / (max_sim - min_sim)) for item, score in sorted_item_list]

    for method, user_recall_items in user_multi_recall_dict.items():
        if not user_recall_items:
            continue
        recall_method_weight = 1.0 if weight_dict is None else weight_dict.get(method, 1.0)
        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in normalize_scores(sorted_item_list):
                final_recall_items_dict[user_id][item] = final_recall_items_dict[user_id].get(item, 0.0) + recall_method_weight * score

    final_recall_items_dict_rank = {
        user: sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        for user, recall_item_dict in final_recall_items_dict.items()
    }

    if save_path:
        save_pickle(final_recall_items_dict_rank, save_path, "final_recall_items_dict.pkl")

    log_step(f"Final merged recall ready for {len(final_recall_items_dict_rank)} users")
    return final_recall_items_dict_rank


def run_multi_recall(
    data_path="./tcdata/",
    save_path="./temp_results/",
    offline=False,
    metric_recall=True,
    enable_methods=None,
    weight_dict=None,
    use_sample=False,
    sample_nums=10000,
):
    enable_methods = set(enable_methods or RECALL_METHODS)
    log_step("Multi-recall pipeline started")
    log_step(f"Enabled methods: {sorted(enable_methods)}")
    artifacts = load_artifacts(
        data_path=data_path,
        save_path=save_path,
        offline=offline,
        use_sample=use_sample,
        sample_nums=sample_nums,
    )
    all_click_df = artifacts["all_click_df"]
    item_info_df = artifacts["item_info_df"]
    item_type_dict = artifacts["item_type_dict"]
    item_words_dict = artifacts["item_words_dict"]
    item_created_time_dict = artifacts["item_created_time_dict"]
    item_emb_df = artifacts["item_emb_df"]

    recall_click_df, trn_last_click_df = get_eval_frames(all_click_df, metric_recall)
    user_multi_recall_dict = {method: {} for method in RECALL_METHODS}

    emb_i2i_sim = {}
    if "embedding_sim_item_recall" in enable_methods or "itemcf_sim_itemcf_recall" in enable_methods or "youtubednn_usercf_recall" in enable_methods:
        emb_i2i_sim, user_multi_recall_dict["embedding_sim_item_recall"] = run_embedding_recall(
            recall_click_df,
            item_created_time_dict,
            item_emb_df,
            save_path,
        )

    if "itemcf_sim_itemcf_recall" in enable_methods:
        user_multi_recall_dict["itemcf_sim_itemcf_recall"] = run_itemcf_recall(
            recall_click_df,
            item_created_time_dict,
            emb_i2i_sim,
            save_path,
        )

    youtube_user_emb_dict = None
    if "youtubednn_recall" in enable_methods or "youtubednn_usercf_recall" in enable_methods:
        user_multi_recall_dict["youtubednn_recall"], youtube_user_emb_dict = youtubednn_u2i_dict(
            recall_click_df,
            item_info_df,
            save_path,
        )

    if "youtubednn_usercf_recall" in enable_methods and youtube_user_emb_dict is not None:
        user_multi_recall_dict["youtubednn_usercf_recall"] = run_youtube_usercf_recall(
            recall_click_df,
            youtube_user_emb_dict,
            item_created_time_dict,
            emb_i2i_sim,
            save_path,
        )

    if "cold_start_recall" in enable_methods:
        all_click_with_info = all_click_df.merge(item_info_df, how="left", on="click_article_id")
        user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(all_click_with_info)
        click_article_ids_set = set(all_click_df["click_article_id"].values)
        source_dict = user_multi_recall_dict["embedding_sim_item_recall"]
        user_multi_recall_dict["cold_start_recall"] = cold_start_items(
            source_dict,
            user_hist_item_typs_dict,
            user_hist_item_ids_dict,
            user_hist_item_words_dict,
            user_last_item_created_time_dict,
            item_type_dict,
            item_words_dict,
            item_created_time_dict,
            click_article_ids_set,
            recall_item_num=100,
        )
        save_pickle(user_multi_recall_dict["cold_start_recall"], save_path, "cold_start_user_items_dict.pkl")

    if metric_recall and trn_last_click_df is not None:
        log_step("Running offline recall metrics for each available method")
        for method, recall_result in user_multi_recall_dict.items():
            if recall_result:
                print(f"\n[{method}]")
                metrics_recall(recall_result, trn_last_click_df, topk=min(50, max(len(items) for items in recall_result.values())))

    final_recall_items_dict = combine_recall_results(
        user_multi_recall_dict,
        weight_dict=weight_dict,
        topk=150,
        save_path=save_path,
    )

    log_step("Multi-recall pipeline finished")
    return user_multi_recall_dict, final_recall_items_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-recall pipeline")
    parser.add_argument("--data-path", default="./tcdata/")
    parser.add_argument("--save-path", default="./results/")
    parser.add_argument("--offline", type=str2bool, default=False, help="Only use train_click_log.csv")
    parser.add_argument("--use-sample", type=str2bool, default=False, help="Use get_all_click_sample instead of get_all_click_df")
    parser.add_argument("--sample-nums", type=int, default=10000, help="Sample user count when --use-sample is enabled")
    parser.add_argument("--metric-recall", type=str2bool, default=True, help="Enable offline recall metrics")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(RECALL_METHODS),
        choices=list(RECALL_METHODS),
        help="Recall methods to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multi_recall(
        data_path=args.data_path,
        save_path=args.save_path,
        offline=args.offline,
        metric_recall=args.metric_recall,
        enable_methods=args.methods,
        use_sample=args.use_sample,
        sample_nums=args.sample_nums,
    )
