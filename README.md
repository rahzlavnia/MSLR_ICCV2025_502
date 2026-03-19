# Skeleton-based Continuous Sign Language Recognition

This repository contains our competition code for the [MSLR 2026 Challenge](https://m-slrt.github.io/MSLR2026/) at CVPR 2026, built upon our ICCV 2025 winning solution. The codebase will be further cleaned up soon.

## Branches

- **`main`** — Original ICCV 2025 solution (3-stream architecture: static/motion/fusion + contrastive learning). Winner (1st place) in both the [Signer-Independent](https://codalab.lisn.upsaclay.fr/competitions/22899) and [Unseen Sentences](https://codalab.lisn.upsaclay.fr/competitions/22900) tasks of [SignEval 2025](https://multimodal-sign-language-recognition.github.io/ICCV-2025/).
- **`cvpr2026`** — MSLR 2026 competition iteration. Simplified to single-stream (static only), upgraded to isharah2000 phase2 dataset, with ensemble inference scripts.

This implementation is largely built upon [VAC](https://github.com/VIPL-SLP/VAC_CSLR) and [CoSign](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html) frameworks.

## Prerequisites

- Pytorch (==2.0.0 recommended for ctcdecode compatibility)
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode), for beam search decoding
- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), for evaluation:
```
mkdir ./software
ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
```

## Setup

1. **Download the dataset** [[download link]](https://www.kaggle.com/competitions/continuous-sign-language-recognition-iccv-2025/data) and place it in `./datasets`.

2. **Download the annotations** [[download link]](https://github.com/gufranSabri/Pose86K-CSLR-Isharah/tree/main/annotations_v2) and place them in `./preprocess/mslr2025`.

3. **Preprocess the dataset**:
```
cd ./preprocess/mslr2025
python mslr_process.py
```

## Pretrained Models (ICCV 2025, `main` branch)

| Task                   | Test WER | Dev WER | Weight |
| ---------------------- | -------- | ------- | ------ |
| **Signer Independent** | 7.44%    | 2.2%    | [Test](https://drive.google.com/file/d/1KMXkr3UG_1Cl2AtCSCUK4eopujPxzqxg/view?usp=drive_link) / [Dev](https://drive.google.com/file/d/1rGc6MqYeEm_JR6AZWweEWrX8e_X26v_p/view?usp=drive_link) |
| **Unseen Sentences**   | 28.20%   | 35.6%   | [Test](https://drive.google.com/file/d/1v9NYOH6ms0DyPcGw1cMjaDGmc_-tBaQ5/view?usp=drive_link) / [Dev](https://drive.google.com/file/d/1ezEFG-xMOyzwpon_XAN_35lV3Tpl9DzW/view?usp=drive_link) |

**Note:** Different tasks benefit from different data augmentation strategies during training. See `./datasets/skeleton_feeder.py` line 194.

## Running

### Signer Independent

```bash
# Train
python main.py --config ./configs/Double_Cosign_si.yaml

# Test
python main.py --config ./configs/Double_Cosign_si.yaml --phase test --load-weights PATH_TO_PRETRAINED_MODEL
```

### Unseen Sentences

```bash
# Train (download pretrained weight from https://drive.google.com/file/d/1oTbcL3gev4DftIFdjahJeMBbLux9Q3Y7/view)
python main.py --config ./configs/Double_Cosign_us.yaml --load-weights PATH_TO_PRETRAINED_MODEL --ignore-weights classifier_static.weight classifier_motion.weight classifier_fusion.weight

# Test
python main.py --config ./configs/Double_Cosign_us.yaml --phase test --load-weights PATH_TO_PRETRAINED_MODEL
```

### Ensemble Inference (`cvpr2026` branch)

```bash
# Ensemble multiple trained models
python ensemble_test.py --model-dirs si_0 si_2 si_4 si_6 --mode dev

# Evaluate ensemble WER
python eval_ensemble_wer.py --ctm PATH_TO_CTM --dataset si --mode dev

# Finetune temporal ensemble
python finetune_temporal_ensemble.py --config ./configs/Double_Cosign_si.yaml --work-dir ./work_dir/finetune/
```

## Citation

```latex
@inproceedings{min2025closer,
  title={A Closer Look at Skeleton-based Continuous Sign Language Recognition},
  author={Min, Yuecong and Yang, Yifan and Jiao, Peiqi and Nan, Zixi and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year={2025}
}

@inproceedings{jiao2023cosign,
  title={Cosign: Exploring co-occurrence signals in skeleton-based continuous sign language recognition},
  author={Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20676--20686},
  year={2023}
}
```
