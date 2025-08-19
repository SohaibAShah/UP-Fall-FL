import numpy as np
import torch
import random

def calculate_relative_loss_reduction_as_list(client_losses):
    """
    计算每个 client 的局部训练损失相对下降幅度 RF_loss，并以列表形式返回。

    参数：
    - client_losses (dict): 一个字典，key 是 client id，value 是一个 list，表示该 client 各轮训练的 loss。

    返回：
    - rf_losses_list (list): 按输入顺序返回每个 client 的 RF_loss。
    """
    # 计算每个 client 的起始损失和结束损失差值
    loss_reductions = {}
    for client_id, losses in client_losses.items():
        if len(losses) < 2:
            raise ValueError(f"Client {client_id} 的训练损失数据不足，至少需要两轮的损失值。")
        loss_start = losses[0]
        loss_end = losses[-1]
        loss_reductions[client_id] = loss_start - loss_end

    # 找到最大损失下降值
    max_loss_reduction = max(loss_reductions.values())

    if max_loss_reduction == 0:
        raise ValueError("所有 client 的损失下降值均为 0，无法计算相对下降幅度。")

    # 计算相对下降幅度，并以列表形式返回
    rf_losses_list = [
        reduction / max_loss_reduction for reduction in loss_reductions.values()
    ]

    return rf_losses_list

def calculate_relative_train_accuracy(client_acc):
    """
    计算每个 client 的局部训练精度 RF_ACC_Train，并返回一个列表。

    参数：
    - client_acc (dict): 一个字典，key 是 client id，value 是该 client 的训练精度。

    返回：
    - rf_acc_train_list (list): 按输入顺序返回每个 client 的 RF_ACC_Train。
    """
    # 找到最大训练精度
    max_acc = max(client_acc.values())

    if max_acc == 0:
        raise ValueError("所有 client 的训练精度均为 0，无法计算相对训练精度。")

    # 计算相对训练精度，并以列表形式返回
    rf_acc_train_list = [
        acc / max_acc for acc in client_acc.values()
    ]

    return rf_acc_train_list


def calculate_global_validation_accuracy(train_acc, global_acc):
    """
    计算每个 client 的全局验证精度 RF_ACC_Global，并返回一个列表。

    参数：
    - train_acc (dict): 一个字典，key 是 client id，value 是该 client 的训练精度。
    - global_acc (dict): 一个字典，key 是 client id，value 是该 client 的全局验证精度。

    返回：
    - rf_acc_global_list (list): 按输入顺序返回每个 client 的 RF_ACC_Global。
    """
    # 检查两个字典是否对齐
    if set(train_acc.keys()) != set(global_acc.keys()):
        raise ValueError("训练精度和全局验证精度的客户端 ID 不一致。")

    # 计算全局验证精度的最大值
    max_global_acc = max(global_acc.values())
    if max_global_acc == 0:
        raise ValueError("所有 client 的全局验证精度均为 0，无法计算 RF_ACC_Global。")

    # 计算全局验证精度与训练精度的差值及其最大值
    global_train_diff = {client_id: global_acc[client_id] - train_acc[client_id] for client_id in train_acc}
    max_global_train_diff = max(global_train_diff.values())
    if max_global_train_diff == 0:
        raise ValueError("所有 client 的全局验证与训练精度差值均为 0，无法计算 RF_ACC_Global。")

    # 计算 RF_ACC_Global
    rf_acc_global_list = [
        (global_acc[client_id] / max_global_acc) - (global_train_diff[client_id] / max_global_train_diff)
        for client_id in train_acc
    ]

    return rf_acc_global_list

def calculate_loss_outliers(client_losses, lambda_loss=1.5):
    """
    计算每个 client 的损失异常 P_loss，并返回一个列表。

    参数：
    - client_losses (dict): 一个字典，key 是 client id，value 是一个包含 j 轮训练损失的列表。
    - lambda_loss (float): 调节标准差影响的系数，默认值为 1.5。

    返回：
    - loss_outliers (list): 按输入顺序返回每个 client 的损失异常分数 P_loss。
    """
    # 提取每个 client 的最终损失
    final_losses = {client_id: losses[-1] for client_id, losses in client_losses.items()}

    # 计算最终损失的均值和标准差
    loss_values = np.array(list(final_losses.values()))
    # loss_values = np.array([value.cpu().numpy() for value in final_losses.values()])
    # loss_values = np.array([value.detach().cpu().numpy() for value in final_losses.values()])

    # 假设 final_losses 是一个字典，其中的值可能是 GPU Tensor
    # loss_values = []
    # for value in final_losses.values():
    #     if isinstance(value, torch.Tensor):
    #         # 检查是否在 GPU 上，并且需要迁移到 CPU
    #         value = value.detach().cpu().numpy() if value.is_cuda else value.detach().numpy()
    #         loss_values.append(value)
    #     else:
    #         raise TypeError(f"Unexpected type {type(value)} in final_losses.values()")

    loss_values = np.array(loss_values)

    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)

    # 计算阈值
    threshold = mean_loss + lambda_loss * std_loss

    # 计算最终损失的最大值
    max_loss = np.max(loss_values)

    # 避免除以零的情况
    if max_loss == 0:
        raise ValueError("所有 client 的最终损失均为 0，无法计算损失异常分数。")

    # 计算损失异常分数
    loss_outliers = [
        final_loss / max_loss if final_loss > threshold else 0
        for final_loss in loss_values
    ]

    return loss_outliers


def calculate_performance_bias(val_acc, global_acc):
    """
    计算每个 client 的性能偏离 P_bias，并返回一个列表。

    参数：
    - val_acc (dict): 一个字典，key 是 client id，value 是该 client 的验证精度。
    - global_acc (dict): 一个字典，key 是 client id，value 是该 client 的全局验证精度。

    返回：
    - performance_bias_list (list): 按输入顺序返回每个 client 的性能偏离值 P_bias。
    """
    # 检查两个字典是否对齐
    if set(val_acc.keys()) != set(global_acc.keys()):
        raise ValueError("验证精度和全局验证精度的客户端 ID 不一致。")

    # 计算性能偏离
    performance_bias_list = []
    for client_id in val_acc:
        val = val_acc[client_id]
        global_val = global_acc[client_id]
        max_val = max(val, global_val)

        # 避免除以零的情况
        if max_val == 0:
            performance_bias = 0  # 如果验证和全局验证精度均为 0，则偏离值为 0
        else:
            performance_bias = abs(val - global_val) / max_val

        performance_bias_list.append(performance_bias)

    return performance_bias_list

def pareto_optimization(
    rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias, client_num
):
    """
    实现 Pareto 优化，筛选节点。

    参数：
    - rf_loss (list): 局部训练损失相对下降幅度。
    - rf_acc_train (list): 局部训练精度。
    - rf_acc_val (list): 局部验证精度。
    - rf_acc_global (list): 全局验证精度。
    - p_loss (list): 损失异常。
    - p_bias (list): 性能偏离。
    - client_num (int): 要选出的节点数。

    返回：
    - selected_clients (list): 选中的 client ID（按输入顺序从 0 开始）。
    """
    # 将输入指标整合为二维数组，便于处理
    # data = np.array([rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, -np.array(p_loss), -np.array(p_bias)]).T

    # 确保所有数组中的元素都转换为 NumPy 数组
    # rf_loss = np.array([x.detach().cpu().numpy() for x in rf_loss])
    rf_loss = np.array(list(rf_loss))
    rf_acc_train = rf_acc_train.detach().cpu().numpy() if isinstance(rf_acc_train, torch.Tensor) else np.array(
        rf_acc_train)
    rf_acc_val = rf_acc_val.detach().cpu().numpy() if isinstance(rf_acc_val, torch.Tensor) else np.array(rf_acc_val)
    rf_acc_global = rf_acc_global.detach().cpu().numpy() if isinstance(rf_acc_global, torch.Tensor) else np.array(
        rf_acc_global)
    p_loss = p_loss.detach().cpu().numpy() if isinstance(p_loss, torch.Tensor) else np.array(p_loss)
    p_bias = p_bias.detach().cpu().numpy() if isinstance(p_bias, torch.Tensor) else np.array(p_bias)
    # rf_acc_train = np.array([x.detach().cpu().numpy() for x in rf_acc_train])
    # rf_acc_val = np.array([x.detach().cpu().numpy() for x in rf_acc_val])
    # rf_acc_global = np.array([x.detach().cpu().numpy() for x in rf_acc_global])
    # p_loss = np.array([x.detach().cpu().numpy() for x in p_loss])
    # p_bias = np.array([x.detach().cpu().numpy() for x in p_bias])

    # 构造 NumPy 数组并转置
    data = np.array([rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, -p_loss, -p_bias]).T

    # Pareto 前沿筛选
    def is_dominated(point, others):
        """判断 point 是否被 others 支配"""
        return any(np.all(other >= point) and np.any(other > point) for other in others)

    pareto_indices = [
        i for i, point in enumerate(data) if not is_dominated(point, np.delete(data, i, axis=0))
    ]
    pareto_clients = pareto_indices

    # 如果前沿节点数多于 client_num，随机选取
    if len(pareto_clients) > client_num:
        return random.sample(pareto_clients, client_num)

    # 如果前沿节点数小于 client_num，基于组合评分补充
    remaining_slots = client_num - len(pareto_clients)
    pareto_scores = [0.4 * rf_loss[i] + 0.6 * rf_acc_global[i] for i in range(len(rf_loss))]
    sorted_indices = np.argsort(pareto_scores)[::-1]  # 按评分从高到低排序

    selected_clients = set(pareto_clients)
    for i in sorted_indices:
        if len(selected_clients) >= client_num:
            break
        if i not in selected_clients:
            selected_clients.add(i)

    return list(selected_clients)

def get_top_clients_with5RF(rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias, client_num):
    # rf_loss = np.array([x.detach().cpu().numpy() for x in rf_loss])
    rf_loss = np.array(list(rf_loss))
    rf_acc_train = rf_acc_train.detach().cpu().numpy() if isinstance(rf_acc_train, torch.Tensor) else np.array(
        rf_acc_train)
    rf_acc_val = rf_acc_val.detach().cpu().numpy() if isinstance(rf_acc_val, torch.Tensor) else np.array(rf_acc_val)
    rf_acc_global = rf_acc_global.detach().cpu().numpy() if isinstance(rf_acc_global, torch.Tensor) else np.array(
        rf_acc_global)
    p_loss = p_loss.detach().cpu().numpy() if isinstance(p_loss, torch.Tensor) else np.array(p_loss)
    p_bias = p_bias.detach().cpu().numpy() if isinstance(p_bias, torch.Tensor) else np.array(p_bias)

    # 计算综合评分
    scores = (
            0.2 * rf_loss +
            0.1 * rf_acc_train +
            0.2 * rf_acc_val +
            0.3 * rf_acc_global -
            0.1 * p_loss -
            0.1 * p_bias
    )
    origin_scores = scores
    # 获取得分最高的前 clientNum 个客户端 ID
    top_client_ids = np.argsort(scores)[::-1][:client_num]  # 降序排序，取前 clientNum 个
    return top_client_ids.tolist(),origin_scores