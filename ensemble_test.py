import argparse
import csv
import glob
import json
import os

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import datasets
import slr_network
import utils
from evaluation.slr_eval.wer_calculation import evaluate
from seq_scripts import write2file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble multiple trained models by averaging probabilities."
    )
    parser.add_argument(
        "--work-dir-root",
        type=str,
        default="./work_dir",
        help="Root folder that contains multiple trained model directories.",
    )
    parser.add_argument(
        "--model-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit model directories. If not set, auto-discover under work-dir-root.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="best_dev*_model.pt",
        help="Glob pattern to select checkpoints inside each model directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Evaluation split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA visible devices, e.g. 0 or 0,1. Set None to run on CPU.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Override test batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./work_dir/ensemble",
        help="Directory to save ensemble outputs.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def discover_model_dirs(work_dir_root, explicit_dirs=None):
    if explicit_dirs:
        dirs = []
        for d in explicit_dirs:
            d = d if os.path.isabs(d) else os.path.join(work_dir_root, d)
            dirs.append(os.path.abspath(d))
    else:
        dirs = []
        root_abs = os.path.abspath(work_dir_root)
        candidates = [root_abs] + [
            os.path.join(root_abs, x)
            for x in os.listdir(root_abs)
            if os.path.isdir(os.path.join(root_abs, x))
        ]
        for d in candidates:
            if os.path.isfile(os.path.join(d, "config.yaml")):
                dirs.append(os.path.abspath(d))
    dirs = sorted(list(set(dirs)))
    if not dirs:
        raise ValueError("No model directory found (missing config.yaml).")
    return dirs


def select_checkpoint(model_dir, pattern):
    ckpts = sorted(glob.glob(os.path.join(model_dir, pattern)))
    if not ckpts:
        # fallback to current checkpoint naming
        ckpts = sorted(glob.glob(os.path.join(model_dir, "cur_dev*_model.pt")))
    if not ckpts:
        raise ValueError(f"No checkpoint found in {model_dir}.")
    # Use latest modified checkpoint to avoid lexical edge cases.
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts[-1]


def load_dataset_info(dataset_name):
    dataset_cfg = f"./configs/dataset_configs/{dataset_name}.yaml"
    if not os.path.isfile(dataset_cfg):
        raise ValueError(f"Dataset config not found: {dataset_cfg}")
    return load_yaml(dataset_cfg)


def build_dataloader(base_cfg, gloss_dict, mode, test_batch_size):
    feeder_name = base_cfg["feeder"]
    feeder_class = getattr(datasets, feeder_name)
    feeder_args = dict(base_cfg["feeder_args"])
    feeder_args["mode"] = mode
    feeder_args["transform_mode"] = False
    feeder_args["dataset"] = base_cfg["dataset"]

    g2i_dict = {k: v["index"] for k, v in gloss_dict["gloss2id"].items()}
    dataset = feeder_class(gloss_dict=g2i_dict, **feeder_args)
    batch_size = (
        test_batch_size
        if test_batch_size is not None
        else int(base_cfg.get("test_batch_size", 4))
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(base_cfg.get("num_worker", 4)),
        collate_fn=feeder_class.collate_fn,
    )
    return loader


def build_model(cfg, gloss_dict, checkpoint_path, output_device):
    model_class = getattr(slr_network, cfg["model"])
    model = model_class(**cfg["model_args"], gloss_dict=gloss_dict)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(output_device)
    model.eval()
    return model


def forward_probs(model, data):
    x, len_x = data["x"], data["len_x"]
    visual_ret = model.visual_module(x, len_x)
    conv1d_module = getattr(model, "conv1d_static")
    contextual_module = getattr(model, "contextual_module_static")
    classifier = getattr(model, "classifier_static")

    conv_logits, seq_logits, feat_len = model.forward_contextual(
        visual_ret, len_x, conv1d_module, contextual_module, classifier
    )
    conv_probs = F.softmax(conv_logits * model.norm_scale, dim=-1)
    seq_probs = F.softmax(seq_logits * model.norm_scale, dim=-1)
    return conv_probs, seq_probs, feat_len


def build_test_csv(ctm_file, csv_path, reference_csv):
    with open(ctm_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    grouped = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        sample_id = parts[0]
        gloss = parts[4]
        grouped.setdefault(sample_id, []).append(gloss)

    ordered_ids = []
    with open(reference_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            sample_id = row[0].strip()
            if not sample_id:
                continue
            if i == 0 and sample_id.lower() == "id":
                continue
            ordered_ids.append(sample_id)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "gloss"])
        for sample_id in ordered_ids:
            words = grouped.get(sample_id, [])
            writer.writerow([sample_id, " ".join(words)])


def check_compatibility(base_cfg, other_cfg, model_dir):
    keys = ["dataset", "feeder", "feeder_args"]
    for k in keys:
        if base_cfg[k] != other_cfg[k]:
            raise ValueError(f"Incompatible config key `{k}` in model dir: {model_dir}")


def main():
    args = parse_args()
    model_dirs = discover_model_dirs(args.work_dir_root, args.model_dirs)

    configs = []
    checkpoints = []
    for model_dir in model_dirs:
        cfg_path = os.path.join(model_dir, "config.yaml")
        cfg = load_yaml(cfg_path)
        ckpt = select_checkpoint(model_dir, args.checkpoint_pattern)
        configs.append(cfg)
        checkpoints.append(ckpt)

    base_cfg = configs[0]
    for cfg, model_dir in zip(configs[1:], model_dirs[1:]):
        check_compatibility(base_cfg, cfg, model_dir)

    dataset_info = load_dataset_info(base_cfg["dataset"])
    with open(dataset_info["dict_path"], "r") as f:
        gloss_dict = json.load(f)

    device = utils.GpuDataParallel()
    if args.device is not None:
        device.set_device(args.device)
    else:
        device.set_device(base_cfg.get("device", "0"))

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    loader = build_dataloader(base_cfg, gloss_dict, args.mode, args.test_batch_size)

    models = []
    for cfg, ckpt in zip(configs, checkpoints):
        model = build_model(cfg, gloss_dict, ckpt, device.output_device)
        models.append(model)
        print(f"[Loaded] {ckpt}")

    decoder = utils.Decode(gloss_dict, len(gloss_dict["id2gloss"]) + 1, "beam")
    total_info = []
    total_sent_fusion = []
    total_sent_conv_fusion = []

    for data in tqdm(loader):
        data = device.dict_data_to_device(data)
        total_info += [item.split("|")[0] for item in data["origin_info"]]

        conv_probs_list = []
        seq_probs_list = []
        feat_len_ref = None
        for model in models:
            with torch.no_grad():
                conv_probs, seq_probs, feat_len = forward_probs(model, data)
            if feat_len_ref is None:
                feat_len_ref = feat_len
            else:
                if not torch.equal(feat_len_ref, feat_len):
                    raise ValueError("Feature lengths mismatch across models.")
            conv_probs_list.append(conv_probs)
            seq_probs_list.append(seq_probs)

        avg_conv_probs = torch.stack(conv_probs_list, dim=0).mean(dim=0)
        avg_seq_probs = torch.stack(seq_probs_list, dim=0).mean(dim=0)

        total_sent_conv_fusion += decoder.decode(
            avg_conv_probs, feat_len_ref, batch_first=False, probs=True
        )
        total_sent_fusion += decoder.decode(
            avg_seq_probs, feat_len_ref, batch_first=False, probs=True
        )

    fusion_ctm = os.path.join(output_dir, f"output-hypothesis-fusion-{args.mode}.ctm")
    conv_ctm = os.path.join(output_dir, f"output-hypothesis-conv-fusion-{args.mode}.ctm")
    write2file(fusion_ctm, total_info, total_sent_fusion)
    write2file(conv_ctm, total_info, total_sent_conv_fusion)
    print(f"[Saved] {fusion_ctm}")
    print(f"[Saved] {conv_ctm}")

    if args.mode == "test":
        test_csv = os.path.join(output_dir, "test.csv")
        setting = base_cfg["feeder_args"].get("setting", "").lower()
        target_ctm = conv_ctm if setting == "us" else fusion_ctm
        build_test_csv(target_ctm, test_csv, "./test.csv")
        print(f"[Saved] {test_csv}")
    else:
        test_csv = os.path.join(output_dir, "test.csv")
        setting = base_cfg["feeder_args"].get("setting", "").lower()
        target_ctm = conv_ctm if setting == "us" else fusion_ctm
        build_test_csv(target_ctm, test_csv, "./test.csv")
        print(f"[Saved] {test_csv}")

        eval_prefix = dataset_info["evaluation_prefix"]
        eval_dir = dataset_info["evaluation_dir"]
        eval_tool = str(base_cfg.get("evaluate_tool", "python")).lower()
        python_eval = eval_tool == "python"

        fusion_wer = evaluate(
            prefix=output_dir + "/",
            mode=args.mode,
            output_file=f"output-hypothesis-fusion-{args.mode}.ctm",
            evaluate_dir=eval_dir,
            evaluate_prefix=eval_prefix,
            output_dir="ensemble_result/",
            python_evaluate=python_eval,
            triplet=True,
        )
        conv_wer = evaluate(
            prefix=output_dir + "/",
            mode=args.mode,
            output_file=f"output-hypothesis-conv-fusion-{args.mode}.ctm",
            evaluate_dir=eval_dir,
            evaluate_prefix=eval_prefix,
            output_dir="ensemble_result/",
            python_evaluate=python_eval,
        )
        print(f"[WER] Conv1D: {conv_wer:.2f}%, BiLSTM: {fusion_wer:.2f}%")
        print(f"[WER] Best: {min(conv_wer, fusion_wer):.2f}%")


if __name__ == "__main__":
    main()
