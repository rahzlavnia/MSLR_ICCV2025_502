import argparse
import os

from evaluation.slr_eval.wer_calculation import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WER for an ensemble CTM output."
    )
    parser.add_argument(
        "--ctm",
        required=True,
        help="Path to hypothesis CTM file (absolute or relative).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["si", "us"],
        help="Dataset split type to select ground-truth prefix.",
    )
    parser.add_argument(
        "--mode",
        default="dev",
        choices=["dev", "test", "train"],
        help="Evaluation mode used to choose ground-truth stm.",
    )
    parser.add_argument(
        "--evaluate-dir",
        default="./evaluation/slr_eval",
        help="Directory containing preprocess/evaluation scripts and stm files.",
    )
    parser.add_argument(
        "--save-detail-dir",
        default="ensemble_eval_result/",
        help="Relative output dir under CTM parent for sclite details.",
    )
    parser.add_argument(
        "--python-evaluate",
        action="store_true",
        help="Use python evaluator (not recommended if pdb breakpoints exist).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ctm_path = os.path.abspath(args.ctm)
    if not os.path.isfile(ctm_path):
        raise FileNotFoundError(f"CTM file not found: {ctm_path}")

    ctm_dir = os.path.dirname(ctm_path) + "/"
    ctm_file = os.path.basename(ctm_path)
    evaluate_prefix = f"mslr-{args.dataset}-groundtruth"

    wer = evaluate(
        prefix=ctm_dir,
        mode=args.mode,
        evaluate_dir=args.evaluate_dir,
        evaluate_prefix=evaluate_prefix,
        output_file=ctm_file,
        output_dir=args.save_detail_dir,
        python_evaluate=args.python_evaluate,
    )
    print(f"CTM: {ctm_path}")
    print(f"Dataset: {args.dataset}, Mode: {args.mode}")
    print(f"Final WER: {wer:.2f}%")


if __name__ == "__main__":
    main()
