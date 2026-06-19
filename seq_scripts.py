import os
import csv
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
from evaluation.slr_eval.wer_calculation import evaluate


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()  
    loss_value = []  
    total_samples = 0  
    total_batches = 0  
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]  

    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)  
        total_batches += 1
        total_samples += len(data['origin_info'])
        ret_dict = model(data)  

        loss, loss_details = model.get_loss(ret_dict, data)  
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data['origin_info'])
            continue
        optimizer.zero_grad()  
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        optimizer.step()  

        loss_value.append(loss.item())  
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.2f}  lr:{clr[0]:.6f}'
            )
            recoder.print_log(
                "\t"
                + ", ".join([f"{k}: {v.item():.2f}" for k, v in loss_details.items()])
            )
    optimizer.scheduler.step()  # Update learning rate scheduler
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))  # Log rata-rata loss
    recoder.print_log(
        f'\tEpoch {epoch_idx} processed {total_samples} samples in {total_batches} batches.'
    )
    return loss_value  # Kembalikan list loss


def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"
):
    model.eval() 
    total_info = [] 
    total_sent_fusion = [] 
    total_sent_conv_fusion = [] 
    
    total_inference_time_wo_decoding = 0.0
    total_inference_time_w_decoding = 0.0
    total_sequences = 0
    
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        data = device.dict_data_to_device(data)  
        batch_sequences = len(data['origin_info'])
        
        data['skip_decoding'] = True # Do not decode during forward pass
        
        with torch.no_grad():
            # W/O Decoding Forward Pass
            start_time_wo = time.time()
            ret_dict = model(data)  # Forward pass tanpa gradien dan tanpa decoding
            end_time_wo = time.time()
            
            # W/ Decoding = W/O Decoding time + Decoding Time
            real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            # Explicit Decoding
            start_time_decoding = time.time()
            conv_sents_fusion = real_model.decoder.decode(
                ret_dict['conv_logits_fusion'] * real_model.norm_scale, ret_dict['feat_len'], batch_first=False, probs=False
            )
            recognized_sents_fusion = real_model.decoder.decode(
                ret_dict['seq_logits_fusion'] * real_model.norm_scale, ret_dict['feat_len'], batch_first=False, probs=False
            )
            end_time_decoding = time.time()

        total_inference_time_wo_decoding += (end_time_wo - start_time_wo)
        total_inference_time_w_decoding += (end_time_wo - start_time_wo) + (end_time_decoding - start_time_decoding)
        total_sequences += batch_sequences

        total_info += [file_name.split("|")[0] for file_name in data['origin_info']]
        total_sent_fusion += recognized_sents_fusion
        total_sent_conv_fusion += conv_sents_fusion

    sps_wo = total_sequences / total_inference_time_wo_decoding if total_inference_time_wo_decoding > 0 else 0
    sps_w = total_sequences / total_inference_time_w_decoding if total_inference_time_w_decoding > 0 else 0
    
    recoder.print_log(f"Inference Speed w/o Decoding : {sps_wo:.2f} seq/s")
    recoder.print_log(f"Inference Speed w/ Decoding  : {sps_w:.2f} seq/s")

    python_eval = True if evaluate_tool == "python" else False


    if mode.startswith('test'):
        results_dir = os.path.join(work_dir, 'test', mode)
    else:
        results_dir = os.path.join(work_dir, 'train', mode)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    write2file(
        os.path.join(results_dir, "output-hypothesis-fusion-{}.ctm".format(mode)), total_info, total_sent_fusion
    )
    write2file(
        os.path.join(results_dir, "output-hypothesis-conv-fusion-{}.ctm".format(mode)), total_info, total_sent_conv_fusion
    )
    
    if mode.startswith('test'):
        csv_file = os.path.join(results_dir, f'{mode}.csv')
        ctm_file = os.path.join(results_dir, "output-hypothesis-fusion-{}.ctm".format(mode))
        with open(ctm_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data = {}
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                id = parts[0]
                word = parts[4]
                if id not in data:
                    data[id] = []
                data[id].append(word)
        data = dict(sorted(data.items(), key=lambda item: item[0]))
        with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "gloss"])
            for id, words in data.items():
                gloss = " ".join(words)
                writer.writerow([id, gloss])

    try:
        lstm_ret_fusion = evaluate(
            prefix=results_dir + "/",
            mode=mode,
            output_file="output-hypothesis-fusion-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=None,
            python_evaluate=python_eval,
            triplet=True,
        )
        conv_ret_fusion = evaluate(
            prefix=results_dir + "/",
            mode=mode,
            output_file="output-hypothesis-conv-fusion-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=None,
            python_evaluate=python_eval,
        )
    except Exception as e:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret_fusion = 100.0
        conv_ret_fusion = 100.0
        
    recoder.print_log(
        f"[{mode.upper()}] Conv1D temporal WER: {conv_ret_fusion: 2.2f}%, BiLSTM contextual WER: {lstm_ret_fusion: 2.2f}%",
        os.path.join(results_dir, f"{mode}_wer.txt")
    )
    return min([conv_ret_fusion, lstm_ret_fusion])


def write2file(path, info, output):
    filereader = open(path, "w") 
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(
                    info[sample_idx], 
                    word_idx * 1.0 / 100, 
                    (word_idx + 1) * 1.0 / 100, 
                    word[0], 
                )
            )
