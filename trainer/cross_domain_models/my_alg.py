import sys
import os
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

sys.path.append("..")
from utils import *
from data.mydataset import data_generator, Load_Dataset, stage_dataset_generator
from models.models import get_backbone_class, Model
from trainer.train_eval import evaluate

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


import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x, y: [batch_size, feature_dim]
        # 计算成本矩阵 C (L2 distance squared)
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        C = torch.sum((torch.abs(x_col - y_lin)) ** 2, 2)

        # 初始化对偶变量
        u = torch.zeros_like(C[:, 0])
        v = torch.zeros_like(C[0, :])

        # Sinkhorn 迭代
        for i in range(self.max_iter):
            u = -self.eps * torch.logsumexp((v.unsqueeze(0) - C) / self.eps, dim=1)
            v = -self.eps * torch.logsumexp((u.unsqueeze(1) - C) / self.eps, dim=0)

        # 计算最优传输计划 P 的对数
        # P_ij = exp((u_i + v_j - C_ij) / eps)
        # Transport cost = sum(P * C)
        log_P = (u.unsqueeze(1) + v.unsqueeze(0) - C) / self.eps
        P = torch.exp(log_P)

        # 防止数值不稳定导致的 NaN
        cost = torch.sum(P * C)

        if self.reduction == 'mean':
            cost = cost / x.size(0)

        return cost


class CoralAlignment(nn.Module):
    def __init__(self):
        super(CoralAlignment, self).__init__()

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # Frobenius norm
        loss = torch.mean(torch.pow(xc - xct, 2))
        # 或者是 sum 也可以，取决于你的 loss scale
        # loss = torch.sum(torch.pow(xc - xct, 2)) / (4*d*d)

        return loss


class MMD_Alignment(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMD_Alignment, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 扩展维度以利用广播机制计算距离矩阵
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 动态计算带宽 (Median Heuristic 的变体)
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 多核加权求和
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # 0. 异常处理：防止空数据导致报错
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)

        batch_size_src = int(source.size()[0])
        batch_size_tgt = int(target.size()[0])

        # 1. 计算联合核矩阵 K (Size: [N+M, N+M])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        # 2. 将核矩阵切分为四个块
        # XX: 源域内部相似度 (左上角)
        XX = kernels[:batch_size_src, :batch_size_src]

        # YY: 目标域内部相似度 (右下角)
        YY = kernels[batch_size_src:, batch_size_src:]

        # XY: 源域-目标域交叉相似度 (右上角)
        XY = kernels[:batch_size_src, batch_size_src:]

        # YX: 目标域-源域交叉相似度 (左下角)，其实等于 XY.T，但在计算 Mean 时可以直接用
        YX = kernels[batch_size_src:, :batch_size_src]

        # 3. 标准 MMD Loss 公式
        # Loss = Mean(XX) + Mean(YY) - 2 * Mean(XY)
        # 注意：这里假设 XY 和 YX 是对称的，所以减去 XY 和 YX 各一次
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        # loss = torch.mean(- XY - YX)
        # print("called!")
        return loss


import torch
import torch.nn as nn


class NCA_MMD_Alignment(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(NCA_MMD_Alignment, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def compute_confidence_weights(self, inputs):
        """
        计算非单调置信度权重 (NCA)
        inputs: tensor shape [N, 3], columns correspond to v, alpha, beta
        """
        # 1. 提取参数 (假设输入拼接顺序为 v, alpha, beta)
        v = inputs[:, 0]
        alpha = inputs[:, 1]
        beta = inputs[:, 2]

        # 2. 计算置信度 Phi
        # 添加 epsilon 防止除零错误
        eps = 1e-6
        phi = 2 * v + alpha + 1.0 / (beta + eps)

        # 3. 计算非单调权重 (高斯形状)
        # 我们希望权重在 Phi 的分布中心(中等置信度)最高
        # mu = phi.mean()  # 中心点
        mu = 18
        # sigma = phi.std() + eps  # 宽度
        sigma = 8

        # 高斯函数: exp(- (x - mu)^2 / (2*sigma^2))
        weights = torch.exp(- (phi - mu) ** 2 / (2 * sigma ** 2))

        # 4. 归一化权重，使其和为 1
        # 这一步对于加权 MMD 很重要，类似于均值计算中的 1/N
        weights = weights / (weights.sum() + eps)

        return weights.detach()

    def gaussian_kernel_weighted(self, source, target, source_weights, target_weights):
        """
        计算加权的高斯核矩阵和
        """
        n_source = int(source.size(0))
        n_target = int(target.size(0))

        # 拼接数据以统一计算距离矩阵
        total = torch.cat([source, target], dim=0)

        # 拼接权重
        total_weights = torch.cat([source_weights, target_weights], dim=0)

        # 计算 L2 距离矩阵
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 计算带宽 bandwidth
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            # 使用中位数或平均距离作为基准带宽
            bandwidth = torch.sum(L2_distance.data) / (total.size(0) ** 2 - total.size(0))

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # 计算多核高斯矩阵
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # 叠加所有核矩阵
        kernel_matrix = sum(kernel_val)  # Shape: [N_total, N_total]

        # --- 核心修改：应用权重 ---
        # 我们需要计算 W * K * W^T
        # weight_matrix[i, j] = w_i * w_j
        weight_matrix = torch.ger(total_weights, total_weights)  # Outer product

        # 加权后的核矩阵
        weighted_kernel_matrix = kernel_matrix * weight_matrix

        # 提取各部分的损失
        # XX: source-source, YY: target-target, XY: source-target
        XX = weighted_kernel_matrix[:n_source, :n_source]
        YY = weighted_kernel_matrix[n_source:, n_source:]
        XY = weighted_kernel_matrix[:n_source, n_source:]
        YX = weighted_kernel_matrix[n_source:, :n_source]

        # MMD Loss = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
        # 注意：因为权重已经归一化，这里直接求和即可，不需要再除以 N^2
        # loss = torch.sum(XX) + torch.sum(YY) - torch.sum(XY) - torch.sum(YX)

        loss = - torch.sum(XY) - torch.sum(YX)

        return loss

    def forward(self, source, target):
        # source, target 形状为 [N, 3] (v, alpha, beta)

        # 1. 计算源域和目标域的非单调权重
        source_weights = self.compute_confidence_weights(source)
        target_weights = self.compute_confidence_weights(target)

        # 2. 计算加权 MMD 损失
        loss = self.gaussian_kernel_weighted(source, target, source_weights, target_weights)

        return loss


def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):

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
    # same_align_criterion = CoralAlignment()
    # same_align_criterion = NCA_MMD_Alignment()
    # same_align_criterion = MMD_Alignment()

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
                # 改进：基于 Phi 的动态决策对齐
                # ============================================================

                # A. 计算置信度 Phi
                # Phi = 2*nu + alpha + 1/beta
                src_phi = compute_model_confidence(src_pos_v, src_pos_alpha, src_pos_beta) # [batch, 1]
                tgt_phi = compute_model_confidence(tgt_v, tgt_alpha, tgt_beta)             # [batch, 1]
                #
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                #
                #
                # # 假设要查看第 3 列 (索引为 2) 的分布
                # column_index = 0
                # column_data = src_phi.to('cpu').detach().numpy()[:, column_index]
                #
                # plt.figure(figsize=(8, 5))
                # # 绘制 KDE 图
                # sns.kdeplot(column_data, fill=True, color='skyblue')
                # plt.title(f'维度数据分布 (KDE) - 第 {column_index} 列')
                # plt.xlabel('数值')
                # plt.ylabel('密度')
                # plt.show()
                #
                # # print(list(tgt_phi.to('cpu').detach().numpy()), file=sys.stderr)
                # print(tgt_phi.shape)

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
                # if batch_reliable_ratio > RELIABLE_RATIO_THRESHOLD:
                if epoch > 0:
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
