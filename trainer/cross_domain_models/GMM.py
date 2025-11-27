import sys
import os
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.append("..")
from utils import *
from data.mydataset import data_generator, Load_Dataset
from trainer.train_eval import evaluate
from models.models import Model


# 辅助函数：提取所有数据的特征和DER
def extract_features_and_der(model, dataloader, device):
    model.eval()
    features_list = []
    der_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs, fea = model(x)
            m, v, alpha, beta = outputs
            # 拼接证据分布参数 DER: [v, alpha, beta]
            der = torch.cat((v, alpha, beta), dim=1)

            features_list.append(fea.cpu())
            der_list.append(der.cpu())
            labels_list.append(y)

    return torch.cat(features_list), torch.cat(der_list), torch.cat(labels_list)


def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):

    dataset_file = torch.load(os.path.join(data_path, f"train_{tgt_id}.pt"))
    tgt_dataset = Load_Dataset(dataset_file, dataset_configs)

    tgt_train_dl = torch.utils.data.DataLoader(dataset=tgt_dataset, batch_size=hparams["batch_size"],
                                               shuffle=True,  # 训练时开启 shuffle
                                               drop_last=True, num_workers=4, pin_memory=True)

    tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")


    dataset_file_src = torch.load(os.path.join(data_path, f"train_{src_id}.pt"))
    src_dataset = Load_Dataset(dataset_file_src, dataset_configs)

    src_loader_full = torch.utils.data.DataLoader(dataset=src_dataset, batch_size=hparams["batch_size"],
                                                  shuffle=False, num_workers=4)


    logger.info('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_EVI_{src_id}.pt')

    # Source Model (Teacher) - 固定
    source_model = Model(dataset_configs, backbone).to(device)
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)

    # Target Model (Student) - 训练
    target_model = Model(dataset_configs, backbone).to(device)
    if hparams['pretrain']:
        target_model.load_state_dict(checkpoint['state_dict'])

    target_encoder = target_model.feature_extractor
    target_encoder.train()
    set_requires_grad(target_encoder, requires_grad=True)
    set_requires_grad(target_model.regressor, requires_grad=False)

    # 优化器与工具
    criterion = RMSELoss()
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(target_optim, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
    scaler = GradScaler()

    best_score, best_loss = 1e10, 1e10
    best_model_wts = copy.deepcopy(target_model.state_dict())

    # -------------------------------------------------------------------------
    # 3. 【核心修改】GMM 初始化与源域中心计算 (含数值稳定修复)
    # -------------------------------------------------------------------------
    logger.info("Initializing GMM and Source Centers...")

    # 3.1 提取源域所有特征和DER
    src_fea_all, src_der_all, src_y_all = extract_features_and_der(source_model, src_loader_full, device)

    # 【关键修复 1】定义并训练标准化器 (StandardScaler)
    # GMM 对特征的尺度非常敏感，必须标准化！
    feat_scaler = StandardScaler()
    src_fea_numpy = src_fea_all.numpy()
    feat_scaler.fit(src_fea_numpy)  # 在源域数据上拟合均值和方差
    src_fea_scaled = feat_scaler.transform(src_fea_numpy)  # 转换源域数据

    # 3.2 训练 GMM
    # 【关键修复 2】reg_covar=1e-3：防止协方差矩阵奇异，这是解决 NaN 的关键参数
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, reg_covar=1e-3)
    gmm.fit(src_fea_scaled)  # 使用标准化后的数据训练

    # 3.3 确定 GMM 组件的物理意义 (排序)
    src_cluster_labels = gmm.predict(src_fea_scaled)
    cluster_ruls = []
    for i in range(3):
        # 防止某个簇为空导致报错
        if (src_cluster_labels == i).sum() > 0:
            mean_rul = src_y_all[src_cluster_labels == i].mean().item()
        else:
            mean_rul = 0.0
        cluster_ruls.append((i, mean_rul))

    # 按 RUL 从大到小排序 (健康 -> 故障)
    sorted_indices = [x[0] for x in sorted(cluster_ruls, key=lambda x: x[1], reverse=True)]
    logger.info(f"GMM Components sorted by RUL (Health->Fault): {sorted_indices}")

    # 3.4 计算源域的“证据中心” (Source DER Centers)
    src_centers = []
    for idx in sorted_indices:
        mask = (src_cluster_labels == idx)
        if mask.sum() > 0:
            center = src_der_all[mask].mean(dim=0)
        else:
            center = torch.zeros(src_der_all.shape[1])  # 兜底
        src_centers.append(center)

    # 转为 Tensor 并放到 GPU
    src_centers = torch.stack(src_centers).to(device)

    # -------------------------------------------------------------------------
    # 4. 训练循环
    # -------------------------------------------------------------------------
    for epoch in range(1, hparams['num_epochs'] + 1):
        target_model.train()  # 确保 encoder 在训练模式
        total_loss = 0
        valid_steps = 0  # 记录有效的 step 数

        for step, (target_x, tgt_y) in enumerate(tgt_train_dl):
            target_x = target_x.to(device, non_blocking=True)
            target_optim.zero_grad()

            with autocast():
                # 1. 目标域前向传播
                tgt_outputs, tgt_fea = target_model(target_x)
                tgt_m, tgt_v, tgt_alpha, tgt_beta = tgt_outputs
                tgt_DER = torch.cat((tgt_v, tgt_alpha, tgt_beta), dim=1)

                # 2. 【关键修复 3】计算隶属度 (Soft Membership)
                # 必须先转 float32，防止 float16 溢出，并转到 CPU
                tgt_fea_cpu = tgt_fea.detach().float().cpu().numpy()

                # 【关键修复 4】使用源域的 scaler 进行标准化
                tgt_fea_scaled = feat_scaler.transform(tgt_fea_cpu)

                # GMM 预测
                probs = gmm.predict_proba(tgt_fea_scaled)  # [batch, 3]

                # 按照之前计算的 sorted_indices 重新排列概率列
                probs = probs[:, sorted_indices]

                # 转回 GPU，加上极小值防止除以0
                membership = torch.tensor(probs, device=device, dtype=torch.float32) + 1e-8

                # 3. 计算软对齐 Loss
                loss = 0
                for k in range(3):
                    # 获取当前 batch 所有样本属于阶段 k 的权重 [batch]
                    weights_k = membership[:, k]

                    # 获取源域对应阶段的中心 [DER_dim]
                    center_k = src_centers[k]

                    # 计算距离: || tgt_DER - src_center_k ||^2
                    dist = torch.sum((tgt_DER - center_k.unsqueeze(0)) ** 2, dim=1)  # [batch]

                    # 加权平均
                    weighted_dist = (dist * weights_k).mean()

                    loss += weighted_dist

            # 【关键修复 5】NaN 熔断机制
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Warning: Loss is {loss} at epoch {epoch} step {step}. Skipping step.")
                scaler.update()  # 即使跳过也要更新 scaler 状态，防止死锁
                continue

            scaler.scale(loss).backward()
            scaler.step(target_optim)
            scaler.update()

            total_loss += loss.item()
            valid_steps += 1

        # 防止除以0
        mean_loss = total_loss / (valid_steps + 1e-6)
        scheduler.step()

        logger.info(f'Epoch: {epoch:02} \t Loss: {mean_loss:.4f}')

        # ---------------------------------------------------------------------
        # 5. 评估与保存
        # ---------------------------------------------------------------------
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,
                                                                 device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)

            if best_score > test_score:
                best_loss, best_score = test_loss, test_score
                best_model_wts = copy.deepcopy(target_model.state_dict())
                logger.info(f'Found new best score: {best_score}, saving model weights...')

            logger.info(f'Src_Only RMSE:{src_only_loss:.4f} \t Score:{src_only_score:.4f}')
            logger.info(f'DA RMSE:{test_loss:.4f} \t Score:{test_score:.4f}')

    # 加载最优模型进行最终测试
    logger.info('Loading best model weights for final evaluation...')
    target_model.load_state_dict(best_model_wts)

    src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs, device)
    test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)

    logger.info(f'Src_Only RMSE:{src_only_loss:.4f} \t Src_Only Score:{src_only_score:.4f}')
    logger.info(f'After DA RMSE:{test_loss:.4f} \t After DA Score:{test_score:.4f}')

    return src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score
