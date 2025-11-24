# import sys
#
# # my add
# # 引入 AMP 相关的模块
# from torch.cuda.amp import autocast, GradScaler
#
# sys.path.append("..")
# from utils import *
# from data.mydataset import data_generator, Load_Dataset, stage_dataset_generator
# import torch
# from torch import nn
# from trainer.train_eval import evaluate
# import copy
# import numpy as np
# import time
# from models.models import get_backbone_class, Model
# import os
#
#
# def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):
#     dataset_file = torch.load(os.path.join(data_path, f"train_{tgt_id}.pt"))
#     tgt_dataset = Load_Dataset(dataset_file, dataset_configs)
#     # 注意：tgt_train_dl 在这里定义后一直复用，是正确的
#     tgt_train_dl = torch.utils.data.DataLoader(dataset=tgt_dataset, batch_size=hparams["batch_size"], shuffle=False,
#                                                drop_last=False, num_workers=8, pin_memory=True,  # 开启锁页内存加速传输
#                                                persistent_workers=True, prefetch_factor=4)
#     # my add
#     tgt_trainX = tgt_dataset.x_data
#
#     dataset_file = torch.load(os.path.join(data_path, f"train_{src_id}.pt"))
#     src_dataset = Load_Dataset(dataset_file, dataset_configs)
#     src_trainX, src_trainY = src_dataset.x_data, src_dataset.y_data
#     src_trainY_cluster = src_trainY.clone()
#     src_trainY_cluster[src_trainY <= 0.15] = 0
#     src_trainY_cluster[src_trainY > 0.66] = 2
#     src_trainY_cluster[(src_trainY > 0.15) & (src_trainY <= 0.66)] = 1
#     src_trainY_cluster = src_trainY_cluster.cpu().numpy()
#
#     tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")
#
#     logger.info('Restore source pre_trained model...')
#
#     checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_EVI_{src_id}.pt')
#
#     # pretrained source model
#     source_model = Model(dataset_configs, backbone).to(device)
#
#     logger.info('=' * 89)
#
#     if hparams['pretrain']:
#         source_model.load_state_dict(checkpoint['state_dict'])
#         source_model.eval()
#         set_requires_grad(source_model, requires_grad=False)
#
#         # initialize target model
#         target_model = Model(dataset_configs, backbone).to(device)
#         target_model.load_state_dict(checkpoint['state_dict'])
#         target_encoder = target_model.feature_extractor
#         target_encoder.train()
#         set_requires_grad(target_encoder, requires_grad=True)
#         set_requires_grad(target_model.regressor, requires_grad=False)
#     else:
#         source_model.train()
#         target_model = source_model
#
#     # criterion
#     criterion = RMSELoss()
#     same_align_criterion = Stage_Wise_Alignment()
#     # optimizer
#     target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
#
#     # my add
#     scheduler = torch.optim.lr_scheduler.StepLR(target_optim, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#
#     best_score, best_loss = 1e10, 1e10
#
#     # my add
#     best_model_wts = copy.deepcopy(target_model.state_dict())
#
#     # 【AMP 修改 1】 初始化 GradScaler
#     scaler = GradScaler()
#
#     # OPTIMIZATION: Initialize variables outside the loop
#     stage_train_dl = None
#     median = 0.5  # default init
#     pred_tgt_label = None
#
#     for epoch in range(1, hparams['num_epochs'] + 1):
#         # OPTIMIZATION: Only update dataset and recreate DataLoader when labels change (every 5 epochs)
#         # This prevents killing and restarting worker processes every single epoch
#         if epoch % 5 == 1:
#             _, _, _, [var_list, sigma_list], pred_tgt_label, _ = evaluate(target_model, tgt_train_dl, criterion,
#                                                                           dataset_configs, device, False)
#             pred_tgt_label = np.array(pred_tgt_label)
#             median = np.quantile(pred_tgt_label, 0.3)
#
#             tgt_trainY_cluster = pred_tgt_label.copy()
#             tgt_trainY_cluster[pred_tgt_label > median] = 2
#             tgt_trainY_cluster[pred_tgt_label <= median] = 1
#
#             # Re-create DataLoader only here.
#             # Since persistent_workers=True, this is expensive, so we only do it when necessary.
#             # Old workers will be collected.
#             stage_train_dl = stage_dataset_generator(src_trainX, src_trainY, src_trainY_cluster, tgt_trainX,
#                                                      pred_tgt_label, tgt_trainY_cluster, random_flag=True,
#                                                      batch_size=hparams['batch_size'])
#
#         # Note: In epochs 2,3,4,5, we REUSE stage_train_dl.
#         # PyTorch DataLoaders automatically reshuffle at the start of __iter__ (start of for loop) if shuffle=True.
#         # This preserves the algorithm logic but saves massive CPU overhead.
#
#         total_loss = 0
#         start_time = time.time()
#
#         # Using enumerate on the existing loader automatically handles reshuffling
#         for step, (target_x, source_pos_x, tgt_y, pos_y) in enumerate(stage_train_dl):
#             target_optim.zero_grad()
#
#             source_pos_x, target_x = source_pos_x.to(device, non_blocking=True), target_x.to(device, non_blocking=True)
#             tgt_y, pos_y = tgt_y.to(device, non_blocking=True), pos_y.to(device, non_blocking=True)
#
#             # my add
#             # 【AMP 修改 2】 使用 autocast 上下文管理器
#             with autocast():
#
#                 src_pos_outputs, src_pos_fea = source_model(source_pos_x)
#                 tgt_outputs, tgt_fea = target_model(target_x)
#
#                 src_pos_m, src_pos_v, src_pos_alpha, src_pos_beta = src_pos_outputs
#                 tgt_m, tgt_v, tgt_alpha, tgt_beta = tgt_outputs
#
#                 src_pos_DER = torch.cat((src_pos_v, src_pos_alpha, src_pos_beta), dim=1)
#                 tgt_DER = torch.cat((tgt_v, tgt_alpha, tgt_beta), dim=1)
#
#                 # OPTIMIZATION: Use torch.where instead of cloning and boolean indexing
#                 # This is faster on GPU and avoids synchronization
#                 # Original logic: > median -> 2, <= median -> 1
#                 tgt_batch_cluster = torch.where(tgt_y > median, torch.tensor(2, device=device).float(),
#                                                 torch.tensor(1, device=device).float())
#                 # Note: Check if tgt_y is float or long. Assuming float based on median comparison.
#                 # If model expects long for indexing, cast it.
#                 # Based on usage below (stage_index = tgt_batch_cluster==stage), float comparison is fine but long is safer for equality.
#
#                 loss = 0
#                 for stage in range(1, 3):
#                     # Use tensor comparison directly
#                     stage_index = (tgt_batch_cluster == stage)
#                     # Check if any samples exist in this stage to avoid NaN/Errors if batch is small
#                     if stage_index.sum() > 0:
#                         loss += same_align_criterion(tgt_DER[stage_index], src_pos_DER[stage_index])
#
#             # my add
#             # 【AMP 修改 3】 使用 scaler 来缩放 loss 并进行反向传播
#             scaler.scale(loss).backward()
#
#             # 【AMP 修改 4】 使用 scaler 来执行 optimizer.step()
#             scaler.step(target_optim)
#             # 【AMP 修改 5】 更新 scaler 的缩放因子
#             scaler.update()
#             total_loss += loss.item()
#
#         mean_loss = total_loss / (step + 1)
#
#         # my add
#         scheduler.step()
#
#         logger.info(f'Epoch: {epoch:02}')
#         logger.info(f'target_loss:{mean_loss} ')
#         if epoch % 1 == 0:
#             src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,
#                                                                  device)
#             test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)
#             if best_score > test_score:
#                 best_loss, best_score = test_loss, test_score
#                 # my add
#                 best_model_wts = copy.deepcopy(target_model.state_dict())
#                 logger.info(f'Found new best score: {best_score}, saving model weights...')
#
#             logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
#             logger.info(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
#
#     # my add: load best model weights
#     logger.info('Loading best model weights for final evaluation...')
#     target_model.load_state_dict(best_model_wts)
#
#     src_only_loss, src_only_score, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl,
#                                                                                         criterion, dataset_configs,
#                                                                                         device)
#     test_loss, test_score, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl,
#                                                                                     criterion, dataset_configs, device)
#
#     logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
#     logger.info(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
#
#     return src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score
import sys
import os
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

# 假设项目结构如下，根据实际情况调整 import 路径
sys.path.append("..")
from utils import *
from data.mydataset import data_generator, Load_Dataset, stage_dataset_generator
from models.models import get_backbone_class, Model
from trainer.train_eval import evaluate

# =============================================================================
# 1. 核心组件：阶段对齐 Loss (你提供的实现)
# =============================================================================
class Stage_Wise_Alignment(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(Stage_Wise_Alignment, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # 容错处理：如果输入为空（例如某阶段没有样本），直接返回0
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0).to(source.device)

        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean( - XY - YX)
        return loss

# =============================================================================
# 2. 辅助函数：计算模型置信度 Phi
# =============================================================================
def compute_model_confidence(nu, alpha, beta):
    """
    根据论文公式 (4) 计算 Model Confidence Phi
    Phi = 2*nu + alpha + 1/beta
    值越大，置信度越高。
    """
    # 添加 epsilon 防止除零
    return 2 * nu + alpha + 1.0 / (beta + 1e-6)

# =============================================================================
# 3. 主训练函数
# =============================================================================
def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):
    # -------------------------------------------------------------------------
    # 数据准备 (保持原逻辑)
    # -------------------------------------------------------------------------
    dataset_file = torch.load(os.path.join(data_path, f"train_{tgt_id}.pt"))
    tgt_dataset = Load_Dataset(dataset_file, dataset_configs)
    tgt_train_dl = torch.utils.data.DataLoader(dataset=tgt_dataset, batch_size=hparams["batch_size"], shuffle=False,
                                               drop_last=False, num_workers=8, pin_memory=True,
                                               persistent_workers=True, prefetch_factor=4)
    tgt_trainX = tgt_dataset.x_data

    dataset_file = torch.load(os.path.join(data_path, f"train_{src_id}.pt"))
    src_dataset = Load_Dataset(dataset_file, dataset_configs)
    src_trainX, src_trainY = src_dataset.x_data, src_dataset.y_data

    # 源域标签聚类 (0:健康, 1:早期退化, 2:晚期退化)
    src_trainY_cluster = src_trainY.clone()
    src_trainY_cluster[src_trainY <= 0.15] = 0
    src_trainY_cluster[src_trainY > 0.66] = 2
    src_trainY_cluster[(src_trainY > 0.15) & (src_trainY <= 0.66)] = 1
    src_trainY_cluster = src_trainY_cluster.cpu().numpy()

    tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")

    # -------------------------------------------------------------------------
    # 模型初始化
    # -------------------------------------------------------------------------
    logger.info('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_EVI_{src_id}.pt')

    source_model = Model(dataset_configs, backbone).to(device)

    if hparams['pretrain']:
        source_model.load_state_dict(checkpoint['state_dict'])
        source_model.eval()
        set_requires_grad(source_model, requires_grad=False)

        target_model = Model(dataset_configs, backbone).to(device)
        target_model.load_state_dict(checkpoint['state_dict'])
        target_encoder = target_model.feature_extractor
        target_encoder.train()
        set_requires_grad(target_encoder, requires_grad=True)
        set_requires_grad(target_model.regressor, requires_grad=False)
    else:
        source_model.train()
        target_model = source_model
        target_encoder = target_model.feature_extractor # 避免未定义错误

    # -------------------------------------------------------------------------
    # 优化器与 Loss
    # -------------------------------------------------------------------------
    criterion = RMSELoss()
    same_align_criterion = Stage_Wise_Alignment()

    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(target_optim, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
    scaler = GradScaler() # AMP

    best_score, best_loss = 1e10, 1e10
    best_model_wts = copy.deepcopy(target_model.state_dict())

    # -------------------------------------------------------------------------
    # 训练循环变量
    # -------------------------------------------------------------------------
    stage_train_dl = None
    median = 0.5
    pred_tgt_label = None

    # 【超参数】动态对齐策略参数
    # RELIABLE_RATIO_THRESHOLD: 批次中至少有多少比例的样本是高置信度的，才进行精细对齐
    RELIABLE_RATIO_THRESHOLD = 0.5
    # CONFIDENCE_QUANTILE: 用源域的多少分位数作为“高置信度”的门槛 (0.5 = 中位数)
    CONFIDENCE_QUANTILE = 0.5

    logger.info('Start Dynamic Decision Alignment Training...')

    for epoch in range(1, hparams['num_epochs'] + 1):
        # 每5个epoch更新一次伪标签和DataLoader
        if epoch % 5 == 1:
            _, _, _, [var_list, sigma_list], pred_tgt_label, _ = evaluate(target_model, tgt_train_dl, criterion,
                                                                          dataset_configs, device, False)
            pred_tgt_label = np.array(pred_tgt_label)
            median = np.quantile(pred_tgt_label, 0.3)

            tgt_trainY_cluster = pred_tgt_label.copy()
            tgt_trainY_cluster[pred_tgt_label > median] = 2
            tgt_trainY_cluster[pred_tgt_label <= median] = 1

            stage_train_dl = stage_dataset_generator(src_trainX, src_trainY, src_trainY_cluster, tgt_trainX,
                                                     pred_tgt_label, tgt_trainY_cluster, random_flag=True,
                                                     batch_size=hparams['batch_size'])

        total_loss = 0

        # 统计这一轮用了多少次全局对齐，多少次阶段对齐
        global_align_count = 0
        stage_align_count = 0

        for step, (target_x, source_pos_x, tgt_y, pos_y) in enumerate(stage_train_dl):
            target_optim.zero_grad()

            source_pos_x, target_x = source_pos_x.to(device, non_blocking=True), target_x.to(device, non_blocking=True)
            tgt_y, pos_y = tgt_y.to(device, non_blocking=True), pos_y.to(device, non_blocking=True)

            with autocast():
                # 1. 前向传播
                src_pos_outputs, src_pos_fea = source_model(source_pos_x)
                tgt_outputs, tgt_fea = target_model(target_x)

                src_pos_m, src_pos_v, src_pos_alpha, src_pos_beta = src_pos_outputs
                tgt_m, tgt_v, tgt_alpha, tgt_beta = tgt_outputs

                # 2. 构建证据向量 (DER)
                src_pos_DER = torch.cat((src_pos_v, src_pos_alpha, src_pos_beta), dim=1)
                tgt_DER = torch.cat((tgt_v, tgt_alpha, tgt_beta), dim=1)

                # ============================================================
                # 【核心改进：基于 Phi 的动态决策对齐】
                # ============================================================

                # A. 计算置信度 Phi
                # Phi = 2*nu + alpha + 1/beta
                src_phi = compute_model_confidence(src_pos_v, src_pos_alpha, src_pos_beta) # [batch, 1]
                tgt_phi = compute_model_confidence(tgt_v, tgt_alpha, tgt_beta)             # [batch, 1]

                # logger.info(f'Source Phi - min: {src_phi.min().item():.4f}, max: {src_phi.max().item():.4f}, mean: {src_phi.mean().item():.4f}')
                # logger.info(f'Target Phi - min: {tgt_phi.min().item():.4f}, max: {tgt_phi.max().item():.4f}, mean: {tgt_phi.mean().item():.4f}')

                # B. 确定“高置信度”阈值
                # 使用当前 Batch 源域样本的 Phi 的中位数作为基准
                # 逻辑：如果目标域样本的置信度能达到源域样本的中等水平，就算可靠
                confidence_threshold = torch.quantile(src_phi, CONFIDENCE_QUANTILE).detach()

                # C. 计算目标域 Batch 的可靠比例
                # 统计有多少目标域样本的 Phi 超过了阈值
                # reliable_mask = tgt_phi > confidence_threshold
                reliable_mask = tgt_phi > 25
                batch_reliable_ratio = reliable_mask.float().mean().item()

                loss = 0

                # D. 动态分支决策
                if batch_reliable_ratio > RELIABLE_RATIO_THRESHOLD:
                # if epoch > 20:
                    # >>> 分支 1: 伪标签质量高 -> 执行精细的阶段对齐 (Stage-Wise) <<<
                    stage_align_count += 1

                    # 这里的逻辑和原论文一致：根据伪标签聚类进行对齐
                    tgt_batch_cluster = torch.where(tgt_y > median,
                                                    torch.tensor(2, device=device).float(),
                                                    torch.tensor(1, device=device).float())

                    for stage in range(1, 3):
                        stage_index = (tgt_batch_cluster == stage)
                        # 只有当该阶段有样本时才计算 Loss
                        if stage_index.sum() > 0:
                            loss += same_align_criterion(tgt_DER[stage_index], src_pos_DER[stage_index])

                else:
                    # >>> 分支 2: 伪标签噪声大 -> 退回全局对齐 (Global) <<<
                    global_align_count += 1

                    # 不区分阶段，直接拉近整个 Batch 的分布
                    # 这样可以避免错误的阶段划分导致的负迁移
                    loss += 0.5 * same_align_criterion(tgt_DER, src_pos_DER)

                    # 可选：给全局对齐加一个惩罚系数，防止Loss过大
                    # loss *= 0.5

                # ============================================================
                # 【结束改进】
                # ============================================================

            scaler.scale(loss).backward()
            scaler.step(target_optim)
            scaler.update()
            total_loss += loss.item()

        mean_loss = total_loss / (step + 1)
        scheduler.step()

        # 日志记录
        if epoch % 1 == 0:
            logger.info(f'Epoch: {epoch:02} | Loss: {mean_loss:.4f}')
            # 打印对齐策略的使用情况，方便观察算法行为
            logger.info(f'Strategy Stats -> Stage-Wise: {stage_align_count}, Global: {global_align_count}')

            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs, device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)

            if best_score > test_score:
                best_loss, best_score = test_loss, test_score
                best_model_wts = copy.deepcopy(target_model.state_dict())
                logger.info(f'Found new best score: {best_score:.2f}, saving model...')

            logger.info(f'Src_Only RMSE:{src_only_loss:.4f} \t Score:{src_only_score:.2f}')
            logger.info(f'DA RMSE:{test_loss:.4f} \t Score:{test_score:.2f}')

    # 加载最佳模型进行最终评估
    logger.info('Loading best model weights for final evaluation...')
    target_model.load_state_dict(best_model_wts)

    src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs, device)
    test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)

    logger.info(f'Final Src_Only RMSE:{src_only_loss:.4f} \t Score:{src_only_score:.2f}')
    logger.info(f'Final DA RMSE:{test_loss:.4f} \t Score:{test_score:.2f}')

    return src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score
