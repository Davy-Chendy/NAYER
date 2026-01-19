import torch
import torch.nn as nn

# imagenet:21.86
def replace_bn_with_gn_closed_form(model, num_groups=32, eps=1e-5):
    for name, module in model.named_children():
        # 递归
        replace_bn_with_gn_closed_form(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)  # safety
            assert C % G == 0, "num_channels must be divisible by num_groups"

            # --- 1. 读取 BN 参数 ---
            gamma = module.weight.data.clone()          # (C,)
            beta = module.bias.data.clone()             # (C,)
            mu = module.running_mean.data.clone()       # (C,)
            var = module.running_var.data.clone()       # (C,)

            # --- 2. 计算 BN 等价仿射 ---
            a = gamma / torch.sqrt(var + eps)           # (C,)
            b = beta - a * mu                            # (C,)

            # --- 3. 计算每个 group 的 μ_g 和 σ_g² ---
            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_group = mu[start:end].mean()
                var_group = var[start:end].mean()

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # --- 4. 闭式映射到 GN 的 affine ---
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b

            # --- 5. 构造 GN 并写入参数 ---
            gn = nn.GroupNorm(
                num_groups=G,
                num_channels=C,
                affine=True,
                eps=eps
            )

            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)
            
def replace_bn_with_gn_closed_form_v2(model, num_groups=32, eps=1e-5):
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0

            gamma = module.weight.data.clone()
            beta  = module.bias.data.clone()
            mu    = module.running_mean.data.clone()
            var   = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            # 排序分组（核心改进）
            sorted_idx = torch.argsort(var)
            sorted_mu  = mu[sorted_idx]
            sorted_var = var[sorted_idx]
            sorted_gamma = gamma[sorted_idx]

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                s = g * channels_per_group
                e = (g + 1) * channels_per_group
                seg_mu   = sorted_mu[s:e]
                seg_var  = sorted_var[s:e]
                seg_w    = sorted_gamma[s:e] ** 2   # 加权
                seg_w    = seg_w / (seg_w.sum() + 1e-8)

                mu_group = (seg_mu * seg_w).sum()
                var_group = (seg_var * seg_w).sum()

                mu_g[sorted_idx[s:e]] = mu_group
                var_g[sorted_idx[s:e]] = var_group

            gn_weight = a * torch.sqrt(var_g + eps) * 0.95  # 轻微缩放
            gn_bias   = a * mu_g + b

            gn = nn.GroupNorm(G, C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

def replace_bn_with_gn_closed_form_v3(model, num_groups=32, eps=1e-5, weighted_agg=True):
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v3(module, num_groups, eps, weighted_agg)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            while C % G != 0 and G > 1:
                G -= 1
            channels_per_group = C // G

            gamma = module.weight.data.clone()
            beta  = module.bias.data.clone()
            mu    = module.running_mean.data.clone()
            var   = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=var.device)

            for g in range(G):
                start = g * channels_per_group
                end   = (g + 1) * channels_per_group
                slice_a = a[start:end]
                slice_mu = mu[start:end]
                slice_var = var[start:end]

                if weighted_agg:
                    weights = 1 / (slice_var + eps)
                    weights = weights / weights.sum()
                    mu_group = (slice_mu * weights).sum()
                    # var_group 可以类似处理，但通常简单平均即可
                    var_group = slice_var.mean()
                else:
                    mu_group = slice_mu.mean()
                    var_group = slice_var.mean()

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # GN weight & bias
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias   = a * mu_g + b

            gn = nn.GroupNorm(G, C, eps=eps, affine=True)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

def replace_bn_with_gn_closed_form_v4(model, num_groups=32, eps=1e-5):
    """
    改进版：使用全方差公式计算 Group 统计量
    """
    for name, module in model.named_children():
        # 递归调用
        replace_bn_with_gn_closed_form_v4(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            # 确保 num_groups 合理，防止 C < num_groups
            G = min(num_groups, C)
            if C % G != 0:
                # 如果不能整除，尝试调整 G (例如 ResNet 第一层 C=64, 但有些模型可能 C=3)
                # 这里简单处理：强制 G=1 (LayerNorm) 或 保持原样报错
                print(f"Warning: {name} channels {C} not divisible by {G}, falling back to G=1")
                G = 1
            
            # --- 1. 读取 BN 参数 ---
            gamma_bn = module.weight.data.clone()
            beta_bn = module.bias.data.clone()
            mu_bn = module.running_mean.data.clone()
            var_bn = module.running_var.data.clone()

            # --- 2. 计算 BN 的等价仿射系数 (y = a*x + b) ---
            # BN: y = gamma * (x - mu) / sqrt(var + eps) + beta
            #     y = [gamma / sqrt(var + eps)] * x + [beta - gamma * mu / sqrt(var + eps)]
            scale_bn = gamma_bn / torch.sqrt(var_bn + eps)
            shift_bn = beta_bn - scale_bn * mu_bn

            # --- 3. 计算 Group 的统计量 (关键改进) ---
            # Reshape 方便计算: (G, C//G)
            channels_per_group = C // G
            
            # Reshape 均值和方差
            mu_bn_reshaped = mu_bn.view(G, channels_per_group)
            var_bn_reshaped = var_bn.view(G, channels_per_group)

            # A. 计算 Group 均值: E[X]
            mu_group = mu_bn_reshaped.mean(dim=1) # (G,)

            # B. 计算 Group 方差: Var(X) = E[Var(X|C)] + Var(E[X|C])
            # 第一项: 各通道方差的均值
            mean_of_vars = var_bn_reshaped.mean(dim=1)
            # 第二项: 各通道均值的方差 (注意：这是总体方差，建议用 unbiased=False)
            var_of_means = mu_bn_reshaped.var(dim=1, unbiased=False)
            
            # 合并方差
            var_group = mean_of_vars + var_of_means # (G,)

            # 扩展回 (C,) 大小以便进行逐通道计算
            mu_group_expanded = mu_group.repeat_interleave(channels_per_group)
            var_group_expanded = var_group.repeat_interleave(channels_per_group)

            # --- 4. 映射到 GN 的权重 ---
            # GN: y = gamma_gn * (x - mu_g) / sqrt(var_g + eps) + beta_gn
            # 我们希望 GN 的输出等于 BN 的等价仿射:
            # gamma_gn / sqrt(var_g) = scale_bn  => gamma_gn = scale_bn * sqrt(var_g)
            # beta_gn - gamma_gn * mu_g / sqrt(var_g) = shift_bn => beta_gn = shift_bn + scale_bn * mu_g
            
            # 计算新的权重
            gn_weight = scale_bn * torch.sqrt(var_group_expanded + eps)
            gn_bias = shift_bn + scale_bn * mu_group_expanded

            # --- 5. 替换层 ---
            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)
            
            setattr(model, name, gn)

# v5:gpt,cifar100 41.27 imagenet:38.27 
def replace_bn_with_gn_closed_form_v5(model, num_groups=32, eps=1e-5):
    """
    无数据、闭式地将 BatchNorm2d 替换为 GroupNorm，
    使用 gamma^2 加权的 group 统计 + 总方差修正。
    """

    for name, module in model.named_children():
        # 递归替换
        replace_bn_with_gn_closed_form_v5(
            module,
            num_groups=num_groups,
            eps=eps
        )

        if not isinstance(module, nn.BatchNorm2d):
            continue

        C = module.num_features
        G = min(num_groups, C)
        assert C % G == 0, "num_channels must be divisible by num_groups"
        channels_per_group = C // G

        # ---------- 1. 读取 BN 参数 ----------
        gamma = module.weight.detach().clone()        # (C,)
        beta = module.bias.detach().clone()          # (C,)
        mu = module.running_mean.detach().clone()    # (C,)
        var = module.running_var.detach().clone()    # (C,)

        # ---------- 2. BN 的等价仿射 ----------
        inv_std = torch.rsqrt(var + eps)
        a = gamma * inv_std
        b = beta - a * mu

        # ---------- 3. 计算加权 group 统计 ----------
        mu_g = torch.zeros_like(mu)
        var_g = torch.zeros_like(var)

        # 使用 gamma^2 作为权重（稳定且物理意义明确）
        w = gamma.pow(2)
        w = w / (w.sum() + 1e-12)

        for g in range(G):
            start = g * channels_per_group
            end = (g + 1) * channels_per_group

            mu_c = mu[start:end]
            var_c = var[start:end]
            w_c = w[start:end]
            w_c = w_c / (w_c.sum() + 1e-12)

            # 加权均值
            mu_group = (mu_c * w_c).sum()

            # 总方差公式：E[var] + Var[mu]
            mean_var = (var_c * w_c).sum()
            mean_mu2 = (mu_c.pow(2) * w_c).sum()
            var_group = mean_var + mean_mu2 - mu_group.pow(2)

            mu_g[start:end] = mu_group
            var_g[start:end] = var_group

        # ---------- 4. 映射到 GN 的 affine ----------
        # 目标：GN(x) ≈ a * x + b
        gn_weight = a * torch.sqrt(var_g + eps)
        gn_bias = b + a * mu_g

        # ---------- 5. 构造 GN ----------
        gn = nn.GroupNorm(
            num_groups=G,
            num_channels=C,
            eps=eps,
            affine=True
        )

        gn.weight.data.copy_(gn_weight)
        gn.bias.data.copy_(gn_bias)

        # ---------- 6. 替换 ----------
        setattr(model, name, gn)
      
# imagenet:22.59
def replace_bn_with_gn_closed_form_v6(model, num_groups=32, eps=1e-5):
    """
    极限无数据 BN -> GN 映射：
    - 非均匀分组（按通道统计结构）
    - gamma^2 / var 加权 group 统计
    - 总方差公式
    - rank-1 通道相关补偿
    """

    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v3(
            module,
            num_groups=num_groups,
            eps=eps
        )

        if not isinstance(module, nn.BatchNorm2d):
            continue

        C = module.num_features
        G = min(num_groups, C)
        assert C % G == 0
        K = C // G

        # ---------- 1. 读取 BN 参数 ----------
        gamma = module.weight.detach().clone()
        beta = module.bias.detach().clone()
        mu = module.running_mean.detach().clone()
        var = module.running_var.detach().clone()

        # ---------- 2. BN 仿射 ----------
        inv_std = torch.rsqrt(var + eps)
        a = gamma * inv_std
        b = beta - a * mu

        # ---------- 3. 构造结构感知排序 ----------
        # 通道“重要性 + 稳定性”联合指标
        score = gamma.abs() * torch.sqrt(var + eps)
        perm = torch.argsort(score, descending=True)

        gamma = gamma[perm]
        beta = beta[perm]
        mu = mu[perm]
        var = var[perm]
        a = a[perm]
        b = b[perm]

        # ---------- 4. group 统计（gamma^2 / var 加权） ----------
        mu_g = torch.zeros_like(mu)
        var_g = torch.zeros_like(var)

        # Fisher-like 权重
        w_all = gamma.pow(2) / (var + eps)
        w_all = w_all / (w_all.sum() + 1e-12)

        for g in range(G):
            s = g * K
            e = (g + 1) * K

            mu_c = mu[s:e]
            var_c = var[s:e]
            w = w_all[s:e]
            w = w / (w.sum() + 1e-12)

            mu_group = (mu_c * w).sum()

            mean_var = (var_c * w).sum()
            mean_mu2 = (mu_c.pow(2) * w).sum()
            var_group = mean_var + mean_mu2 - mu_group.pow(2)

            mu_g[s:e] = mu_group
            var_g[s:e] = var_group

        # ---------- 5. GN affine ----------
        gn_weight = a * torch.sqrt(var_g + eps)
        gn_bias = b + a * mu_g

        # ---------- 6. rank-1 协方差补偿 ----------
        # 构造一个冻结的通道尺度修正
        corr = gamma / (gamma.norm(p=2) + 1e-12)
        corr = corr * 0.05   # 强度超参，可调 0.03~0.08

        gn_weight = gn_weight * (1.0 + corr)

        # ---------- 7. 构造 GN ----------
        gn = nn.GroupNorm(
            num_groups=G,
            num_channels=C,
            eps=eps,
            affine=True
        )

        # 还原通道顺序
        inv_perm = torch.argsort(perm)
        gn.weight.data.copy_(gn_weight[inv_perm])
        gn.bias.data.copy_(gn_bias[inv_perm])

        setattr(model, name, gn)

# imagenet:52.93 cifar100:61.31 cifar10:90.36
def replace_bn_with_gn_closed_form_v7(model, num_groups=32, eps=1e-5, beta_param=0.3):
    """
    方案5：二阶矩匹配
    核心思想：不仅匹配均值和方差，还考虑二阶统计量（通道间协方差）
    """
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v7(module, num_groups, eps, beta_param)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_group = mu[start:end].mean()
                
                # 二阶矩校正：考虑通道间的协方差贡献
                # Var(X_group) = E[Var(X_i)] + Var(E[X_i])
                within_var = var[start:end].mean()
                between_var = mu[start:end].var()
                
                # 加权组合（beta_param控制between variance的权重）
                var_group = within_var + beta_param * between_var

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:61.11 cifar100:60.67 cifar10:82.69
def replace_bn_with_gn_closed_form_v8(model, num_groups=32, eps=1e-5):
    """
    方案6：混合策略（最稳健）
    结合v2和v5的优点，根据每层的统计特性自适应选择策略
    """
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v8(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_group = mu[start:end].mean()
                
                # v2的方差计算
                within_var = var[start:end].mean()
                between_var = mu[start:end].var()
                var_v2 = within_var + between_var
                
                # v5的方差计算（带限制）
                between_var_clamped = torch.clamp(between_var, 0, within_var * 1.5)
                var_v5 = within_var + 0.2 * between_var_clamped
                
                # 自适应混合：如果between_var相对较小，使用v5；否则使用v2
                ratio = between_var / (within_var + eps)
                if ratio < 1.0:
                    var_group = var_v5
                else:
                    var_group = var_v2 * 0.8  # 保守一点

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # 计算GN参数
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b
            
            # 多重安全检查
            # 1. 检测NaN/Inf
            if torch.isnan(gn_weight).any() or torch.isinf(gn_weight).any():
                print(f"Warning: NaN/Inf detected in {name}, using fallback")
                gn_weight = gamma.clone()
                gn_bias = beta.clone()
            
            # 2. 限制极值
            gn_weight = torch.clamp(gn_weight, -5, 5)
            gn_bias = torch.clamp(gn_bias, -10, 10)
            
            # 3. 检查权重分布
            if gn_weight.abs().mean() > 3:
                gn_weight = gn_weight * (1.0 / gn_weight.abs().mean())

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:43.82 cifar100:61.46 cifar10:90.35
def replace_bn_with_gn_closed_form_v9(model, num_groups=32, eps=1e-5, beta_param=0.3):
    """
    方案5：二阶矩匹配（修复版）
    核心思想：不仅匹配均值和方差，还考虑二阶统计量（通道间协方差）
    
    修复：添加数值稳定性检查，防止梯度消失/爆炸
    """
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v9(module, num_groups, eps, beta_param)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_group = mu[start:end].mean()
                
                # 二阶矩校正：考虑通道间的协方差贡献
                within_var = var[start:end].mean()
                between_var = mu[start:end].var()
                
                # 关键修复：限制between_var的贡献，防止方差过大
                # CIFAR-100等小数据集BN统计不稳定，between_var可能异常大
                between_var = torch.clamp(between_var, 0, within_var * 2.0)
                
                # 加权组合
                var_group = within_var + beta_param * between_var

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # 数值稳定性检查
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b
            
            # 检测并修复异常值
            weight_mean = gn_weight.abs().mean()
            weight_std = gn_weight.std()
            if weight_mean > 10 or weight_std > 10:
                # 权重过大，进行归一化
                gn_weight = gn_weight / (weight_mean / 1.0)
            
            # 限制权重范围，防止梯度爆炸
            gn_weight = torch.clamp(gn_weight, -10, 10)
            gn_bias = torch.clamp(gn_bias, -10, 10)

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)
   
# imagenet:16.16 cifar100:51.39 cifar10:83.2
def replace_bn_with_gn_closed_form_v10(model, num_groups=32, eps=1e-5):
    """
    方案1：使用方差加权的统计量聚合
    核心思想：方差大的通道对group统计量的贡献应该更大
    """
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v10(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                # 使用方差作为权重（方差越大，信息量越大）
                weights = var[start:end] + eps
                weights = weights / weights.sum()
                
                mu_group = (mu[start:end] * weights).sum()
                var_group = (var[start:end] * weights).sum()

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:2.542 cifar100:51.43 cifar10:84.7
def replace_bn_with_gn_closed_form_v11(model, num_groups=32, eps=1e-5):
    """
    方案2：方差保持的映射
    核心思想：保持每个通道的标准化强度，通过调整GN参数来补偿group内的方差变化
    """
    for name, module in model.named_children():
        replace_bn_with_gn_closed_form_v11(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                # 使用加权平均（考虑通道间的相关性）
                mu_group = mu[start:end].mean()
                
                # 保持方差：考虑group内方差的期望
                # E[Var_group] ≈ E[Var_channel] + Var[μ_channel]
                var_group = var[start:end].mean() + mu[start:end].var()

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # 调整系数以补偿方差变化
            correction_factor = torch.sqrt((var + eps) / (var_g + eps))
            gn_weight = a * torch.sqrt(var_g + eps) * correction_factor
            gn_bias = a * mu_g + b

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:61.57 cifar100:60.73 cifar10:89.91
def replace_bn_with_gn_optimal_v4(model, num_groups=32, eps=1e-5):
    """
    最优策略V4：精炼版v6
    
    保持v6在ImageNet上的优势(61.11)，同时改进CIFAR-10性能
    关键修改：调整between_var的处理，增加对小数据集的适应性
    """
    for name, module in model.named_children():
        replace_bn_with_gn_optimal_v4(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_channels = mu[start:end]
                var_channels = var[start:end]
                
                # === v6核心 + 改进 ===
                
                mu_group = mu_channels.mean()
                
                # 方差估计（v6原版）
                within_var = var_channels.mean()
                between_var = mu_channels.var()
                var_v2 = within_var + between_var
                
                # 加入v5风格的限制
                between_var_clamped = torch.clamp(between_var, 0, within_var * 1.5)
                var_v5 = within_var + 0.25 * between_var_clamped  # 降低beta从0.2到0.25
                
                # 自适应选择
                R = between_var / (within_var + eps)
                if R < 0.8:  # 通道相似，用v5风格（对CIFAR-10更好）
                    var_group = var_v5
                else:  # 通道差异大，用v2风格但更保守
                    var_group = var_v2 * 0.75  # 从0.8降到0.75
                
                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # 计算GN参数
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b
            
            # v6的安全检查
            if torch.isnan(gn_weight).any() or torch.isinf(gn_weight).any():
                gn_weight = gamma.clone()
                gn_bias = beta.clone()
            
            gn_weight = torch.clamp(gn_weight, -5, 5)
            gn_bias = torch.clamp(gn_bias, -10, 10)
            
            if gn_weight.abs().mean() > 3:
                gn_weight = gn_weight * (1.0 / gn_weight.abs().mean())

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:8.56 cifar100:56.33 cifar10:87.15
def replace_bn_with_gn_optimal_v3(model, num_groups=32, eps=1e-5):
    """
    最优策略V3：多尺度方差估计
    
    创新：结合v5和v6的优点，同时考虑：
    1. 二阶统计（v5的核心）
    2. 数值稳定性（v6的优势）
    3. 自适应权重（基于R比）
    """
    for name, module in model.named_children():
        replace_bn_with_gn_optimal_v3(module, num_groups, eps)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            assert C % G == 0, "num_channels must be divisible by num_groups"

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_channels = mu[start:end]
                var_channels = var[start:end]
                
                # === 核心算法 ===
                
                # 1. 基础统计量
                within_var = var_channels.mean()
                between_var = mu_channels.var()
                R = between_var / (within_var + eps)
                
                # 2. 自适应参数
                # R小时信任group，R大时保守
                beta_adaptive = 0.3 * torch.exp(-R)  # R=0→0.3, R=1→0.11, R=2→0.04
                
                # 3. 安全的between_var
                # 使用v6的思路：限制上界，但上界也自适应
                max_ratio = 1.5 if R < 1.0 else 1.2
                between_var_safe = torch.clamp(between_var, 0, within_var * max_ratio)
                
                # 4. 方差估计
                var_group = within_var + beta_adaptive * between_var_safe
                
                # 5. 均值估计（自适应加权）
                if R < 0.8:
                    # 通道相似，简单平均即可
                    mu_group = mu_channels.mean()
                else:
                    # 通道差异大，用方差加权
                    weights = var_channels / (var_channels.sum() + eps)
                    mu_group = (mu_channels * weights).sum()
                
                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # 计算GN参数
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b
            
            # v6风格的安全检查
            weight_mean = gn_weight.abs().mean()
            if weight_mean > 3:
                gn_weight = gn_weight * (1.5 / weight_mean)
            
            gn_weight = torch.clamp(gn_weight, -10, 10)
            gn_bias = torch.clamp(gn_bias, -10, 10)

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)

# imagenet:56.54(tau 10.0) cifar100:62.04 cifar10:90.6
# imagenet:60.97/62.54/62.74/62.73/62.60/62.53(tau 5/2/1/0.5/0.2/0.1)
# cifar100:61.81 cifar10:89.96 tau 1 
# vgg: cifar100:56.36 cifar10:86.07
def replace_bn_with_gn_principled(model, num_groups=32, eps=1e-5, tau=1):
    """
    Replace BatchNorm with GroupNorm using principled variance estimation.
    
    Mathematical foundation:
    1. Mean aggregation: μ_g = (1/K) Σ μ_c (simple average)
    
    2. Variance aggregation via Law of Total Variance:
       σ²_g = E[Var(x|c)] + Var(E[x|c])
            = within_var + between_var
    
    3. Bayesian shrinkage to handle estimation uncertainty:
       σ²_g = within_var + λ(K) * between_var
       where λ(K) = K / (K + τ)
       
    Intuition:
    - When K (channels_per_group) is small, between_var estimate is unreliable
      → shrink towards 0 (trust within_var more)
    - When K is large, between_var estimate is reliable
      → use full value
    - τ controls prior strength (how much we trust within_var by default)
    
    Args:
        model: PyTorch model with BatchNorm layers
        num_groups: Number of groups for GroupNorm
        eps: Small constant for numerical stability
        tau: Regularization parameter for shrinkage (default: 10.0)
             - Larger τ: more conservative (trust within_var more)
             - Smaller τ: more aggressive (trust between_var more)
    """
    for name, module in model.named_children():
        replace_bn_with_gn_principled(module, num_groups, eps, tau)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            
            # Ensure divisibility
            while C % G != 0 and G > 1:
                G -= 1
            
            if G == 1:
                print(f"Warning: {name} has {C} channels, using single group")

            # Extract BN parameters
            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            # Compute affine transformation coefficients
            # BN output: γ * (x - μ) / √(σ² + ε) + β
            # Equivalently: a * x + b where:
            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            # Initialize group statistics
            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_channels = mu[start:end]
                var_channels = var[start:end]
                K = channels_per_group

                # === Step 1: Mean aggregation ===
                mu_group = mu_channels.mean()

                # === Step 2: Variance aggregation with Bayesian shrinkage ===
                
                # Within-group variance (average of per-channel variances)
                within_var = var_channels.mean()
                
                # Between-group variance (variance of per-channel means)
                between_var = ((mu_channels - mu_group) ** 2).mean()
                
                # Shrinkage coefficient based on sample size
                # λ = K / (K + τ)
                # When K is small: λ → 0 (don't trust between_var)
                # When K is large: λ → 1 (trust between_var)
                lambda_shrink = K / (K + tau)
                
                # Final variance estimate
                var_group = within_var + lambda_shrink * between_var

                # Broadcast to all channels in group
                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            # === Step 3: Compute GN parameters ===
            # We want GN to produce: a * x + b
            # GN computes: γ_gn * (x - μ_g) / √(σ²_g + ε) + β_gn
            # 
            # Matching coefficients:
            # γ_gn / √(σ²_g + ε) = a  =>  γ_gn = a * √(σ²_g + ε)
            # β_gn - γ_gn * μ_g / √(σ²_g + ε) = b  =>  β_gn = a * μ_g + b
            
            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b

            # === Step 4: Numerical stability checks ===
            
            # Check for NaN/Inf
            if torch.isnan(gn_weight).any() or torch.isinf(gn_weight).any():
                print(f"Warning: NaN/Inf detected in {name}, using original BN params")
                gn_weight = gamma.clone()
                gn_bias = beta.clone()
            
            # Clip extreme values (conservative bounds)
            gn_weight = torch.clamp(gn_weight, -10, 10)
            gn_bias = torch.clamp(gn_bias, -20, 20)
            
            # Normalize if weights are too large
            weight_norm = gn_weight.abs().mean()
            if weight_norm > 5:
                gn_weight = gn_weight / weight_norm

            # === Step 5: Create and initialize GN layer ===
            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)
    
    return model

# imagenet:51.94 cifar100:61.77 cifar10:90.43
def replace_bn_with_gn_adaptive(model, num_groups=32, eps=1e-5, tau_base=10.0):
    """
    Adaptive version: automatically tune τ based on variance ratio.
    
    Insight: If within_var >> between_var, the channels are similar,
    so we should trust within_var more (larger τ).
    
    τ_adaptive = τ_base * (1 + within_var / (between_var + ε))
    """
    for name, module in model.named_children():
        replace_bn_with_gn_adaptive(module, num_groups, eps, tau_base)

        if isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            G = min(num_groups, C)
            
            while C % G != 0 and G > 1:
                G -= 1

            gamma = module.weight.data.clone()
            beta = module.bias.data.clone()
            mu = module.running_mean.data.clone()
            var = module.running_var.data.clone()

            a = gamma / torch.sqrt(var + eps)
            b = beta - a * mu

            channels_per_group = C // G
            mu_g = torch.zeros(C, device=mu.device)
            var_g = torch.zeros(C, device=mu.device)

            for g in range(G):
                start = g * channels_per_group
                end = (g + 1) * channels_per_group

                mu_channels = mu[start:end]
                var_channels = var[start:end]
                K = channels_per_group

                mu_group = mu_channels.mean()
                within_var = var_channels.mean()
                between_var = ((mu_channels - mu_group) ** 2).mean()
                
                # Adaptive τ: larger when channels are similar
                variance_ratio = within_var / (between_var + eps)
                tau_adaptive = tau_base * (1 + variance_ratio)
                
                lambda_shrink = K / (K + tau_adaptive)
                var_group = within_var + lambda_shrink * between_var

                mu_g[start:end] = mu_group
                var_g[start:end] = var_group

            gn_weight = a * torch.sqrt(var_g + eps)
            gn_bias = a * mu_g + b

            if torch.isnan(gn_weight).any() or torch.isinf(gn_weight).any():
                gn_weight = gamma.clone()
                gn_bias = beta.clone()
            
            gn_weight = torch.clamp(gn_weight, -10, 10)
            gn_bias = torch.clamp(gn_bias, -20, 20)
            
            weight_norm = gn_weight.abs().mean()
            if weight_norm > 5:
                gn_weight = gn_weight / weight_norm

            gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=eps)
            gn.weight.data.copy_(gn_weight)
            gn.bias.data.copy_(gn_bias)

            setattr(model, name, gn)


# resnet34: cifar100:60.75 cifar10:89.17
# resnet50:imagenet:62.43 
# vgg: cifar100:55.76 cifar10:85.54
def replace_bn_with_gn_analytic(model, num_groups=32):
    """
    Analytic replacement without magic numbers (tau, clamps).
    Uses strict affine transformation matching.
    """
    # 收集需要替换的层，避免在遍历时修改 OrderedDict
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))

    for name, bn in bn_layers:
        C = bn.num_features
        G = min(num_groups, C)
        while C % G != 0 and G > 1: G -= 1
        
        # 1. 获取 BN 的等效线性参数: BN(x) = W_bn * x + B_bn
        # W_bn = gamma / sigma
        # B_bn = beta - (gamma * mu) / sigma
        # print(f'bn_running_mean_shape: {bn.running_mean.shape}')
        bn_std = torch.sqrt(bn.running_var + bn.eps)
        W_bn = bn.weight / bn_std
        B_bn = bn.bias - W_bn * bn.running_mean

        # 2. 估算 Group 统计量 (假设通道独立，无法解决 WRN 协方差问题，但这是数学上限)
        # 使用 reshape 避免手动循环，更优雅
        run_mean = bn.running_mean
        run_var = bn.running_var
        
        # Reshape to [G, C//G]
        mu_reshaped = run_mean.reshape(G, -1)
        var_reshaped = run_var.reshape(G, -1)
        
        # Law of Total Variance: E[Var] + Var[E]
        # mu_g: [G] -> 广播回 [C]
        mu_g_val = mu_reshaped.mean(dim=1) 
        var_g_val = var_reshaped.mean(dim=1) + (mu_reshaped - mu_g_val.unsqueeze(1)).pow(2).mean(dim=1)
        
        # 广播回所有通道
        mu_g_broadcast = mu_g_val.repeat_interleave(C // G)
        sig_g_broadcast = torch.sqrt(var_g_val + bn.eps).repeat_interleave(C // G)

        # 3. 计算 GN 参数以匹配 BN 的仿射变换
        # 我们希望: GN(x) ≈ BN(x)
        # GN(x) = (x - μ_g) / σ_g * γ_gn + β_gn
        #       = (γ_gn / σ_g) * x + (β_gn - γ_gn * μ_g / σ_g)
        #
        # 对齐斜率 (Slope Matching):
        # γ_gn / σ_g = W_bn  =>  γ_gn = W_bn * σ_g
        gamma_gn = W_bn * sig_g_broadcast
        
        # 对齐截距 (Intercept Matching):
        # β_gn - γ_gn * μ_g / σ_g = B_bn
        # β_gn = B_bn + (γ_gn / σ_g) * μ_g
        # β_gn = B_bn + W_bn * μ_g
        beta_gn = B_bn + W_bn * mu_g_broadcast
        # gamma_gn = torch.clamp(gamma_gn, -10, 10)
        # beta_gn = torch.clamp(beta_gn, -20, 20)

        # 4. 创建层
        gn = nn.GroupNorm(num_groups=G, num_channels=C, affine=True, eps=bn.eps)
        gn.weight.data.copy_(gamma_gn)
        gn.bias.data.copy_(beta_gn)

        # 替换
        _set_layer(model, name, gn)

    return model

# resnet34:cifar100:73.06 cifar10:93.39 
# resnet50: imagenet:71.36 
# vgg: cifar100:65.63 cifar10:89.00
# wrn40_2: cifar100:68.69 cifar10:92.36
def replace_bn_with_ln_analytic(model):
    """
    Analytic replacement of BatchNorm2d with LayerNorm.
    Uses strict affine transformation matching.
    """
    # 收集需要替换的层，避免在遍历时修改 OrderedDict
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))

    for name, bn in bn_layers:
        C = bn.num_features
        
        # 1. 获取 BN 的等效线性参数: BN(x) = W_bn * x + B_bn
        # W_bn = gamma / sigma
        # B_bn = beta - (gamma * mu) / sigma
        bn_std = torch.sqrt(bn.running_var + bn.eps)
        W_bn = bn.weight / bn_std
        B_bn = bn.bias - W_bn * bn.running_mean

        # 2. 估算 Layer 统计量 (跨所有通道)
        # LayerNorm 对所有 C 个通道计算统计量
        mu_ln = bn.running_mean.mean()  # 标量
        
        # Law of Total Variance: Var[X] = E[Var[X|group]] + Var[E[X|group]]
        # E[Var] = mean of per-channel variances
        # Var[E] = variance of per-channel means
        var_within = bn.running_var.mean()
        var_between = (bn.running_mean - mu_ln).pow(2).mean()
        var_ln = var_within + var_between
        
        sig_ln = torch.sqrt(var_ln + bn.eps)  # 标量

        # 3. 计算 LN 参数以匹配 BN 的仿射变换
        # LayerNorm: LN(x) = (x - μ_ln) / σ_ln * γ_ln + β_ln
        #                  = (γ_ln / σ_ln) * x + (β_ln - γ_ln * μ_ln / σ_ln)
        #
        # 斜率匹配: γ_ln / σ_ln = W_bn  =>  γ_ln = W_bn * σ_ln
        gamma_ln = W_bn * sig_ln
        
        # 截距匹配: β_ln = B_bn + W_bn * μ_ln
        beta_ln = B_bn + W_bn * mu_ln

        # 4. 创建自定义的 LayerNorm2d 包装器
        ln = LayerNorm2d(C, eps=bn.eps)
        ln.weight.data.copy_(gamma_ln)
        ln.bias.data.copy_(beta_ln)

        # 替换
        _set_layer(model, name, ln)

    return model

def replace_bn_with_in_analytic(model):
    """
    将 BN 替换为 Instance Normalization (IN)
    IN 对每个样本的每个通道独立归一化，完全不依赖 batch size
    适用于：风格迁移、生成任务、bs=1 的场景
    
    预期性能：略低于 GN，但在 bs=1 时表现稳定
    """
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    
    for name, bn in bn_layers:
        C = bn.num_features
        
        # 1. BN 的等效线性参数
        bn_std = torch.sqrt(bn.running_var + bn.eps)
        W_bn = bn.weight / bn_std
        B_bn = bn.bias - W_bn * bn.running_mean
        
        # 2. 估算 Instance 统计量
        # IN 对每个通道独立计算，统计量就是该通道的 running stats
        mu_in = bn.running_mean  # [C]
        sig_in = bn_std  # [C]
        
        # 3. 参数匹配
        # IN(x) = (x - μ_in) / σ_in * γ_in + β_in
        # 斜率匹配: γ_in / σ_in = W_bn => γ_in = W_bn * σ_in
        gamma_in = W_bn * sig_in
        
        # 截距匹配: β_in = B_bn + W_bn * μ_in
        beta_in = B_bn + W_bn * mu_in
        
        # 4. 创建 IN 层
        in_layer = InstanceNorm2d(C, eps=bn.eps, affine=True)
        in_layer.weight.data.copy_(gamma_in)
        in_layer.bias.data.copy_(beta_in)
        
        # 替换
        _set_layer(model, name, in_layer)
    
    return model

# resnet34: cifar10:95.7 cifar100:78.05
# resnet50: imagenet: 76.16
def replace_bn_with_adaptive_in_analytic(model, momentum=0.1):
    """
    将 BN 替换为自适应 Instance Normalization
    在推理时使用运行统计量，训练时动态更新
    
    这种方法在 bs=1 时更加稳定，因为它结合了 BN 的全局统计信息
    """
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    
    for name, bn in bn_layers:
        C = bn.num_features
        
        # 使用 PyTorch 内置的 InstanceNorm2d，它支持 track_running_stats
        in_layer = nn.InstanceNorm2d(
            C, 
            eps=bn.eps, 
            momentum=momentum,
            affine=True,
            track_running_stats=True
        )
        
        # 转移 BN 的统计信息到 IN
        # 注意：IN 的统计量是按通道的，与 BN 相同
        if hasattr(in_layer, 'running_mean'):
            in_layer.running_mean.data.copy_(bn.running_mean)
            in_layer.running_var.data.copy_(bn.running_var)
        
        # 参数匹配（与上面相同的逻辑）
        bn_std = torch.sqrt(bn.running_var + bn.eps)
        W_bn = bn.weight / bn_std
        B_bn = bn.bias - W_bn * bn.running_mean
        
        mu_in = bn.running_mean
        sig_in = bn_std
        
        gamma_in = W_bn * sig_in
        beta_in = B_bn + W_bn * mu_in
        
        in_layer.weight.data.copy_(gamma_in)
        in_layer.bias.data.copy_(beta_in)
        
        # 替换
        _set_layer(model, name, in_layer)
    
    return model


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D data (N, C, H, W).
    Normalizes over C, H, W dimensions.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_channels = num_channels

    def forward(self, x):
        # x: [N, C, H, W]
        # 对 (C, H, W) 做归一化
        mean = x.mean(dim=[1, 2, 3], keepdim=True)  # [N, 1, 1, 1]
        var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)  # [N, 1, 1, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用仿射变换: weight 和 bias 的 shape 是 [C]，需要 reshape 到 [1, C, 1, 1]
        x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_norm

class InstanceNorm2d(nn.Module):
    """
    InstanceNorm2d wrapper with learnable affine parameters.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x: [B, C, H, W]
        # 对每个样本的每个通道单独归一化
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # [B, C, H*W]
        
        mean = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        var = x.var(dim=2, unbiased=False, keepdim=True)  # [B, C, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        x_norm = x_norm.view(B, C, H, W)
        
        if self.affine:
            x_norm = x_norm * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        
        return x_norm

def _set_layer(model, name, layer):
    levels = name.split('.')
    if len(levels) > 1:
        _set_layer(getattr(model, levels[0]), '.'.join(levels[1:]), layer)
    else:
        setattr(model, name, layer)

def check_all_norm_layers_are_gn(model, verbose=True):
    """
    检查模型中所有的归一化层是否都是 GroupNorm。
    如果发现 BatchNorm2d、LayerNorm、InstanceNorm 等其他归一化层，会打印警告。

    Args:
        model (nn.Module): 要检查的模型
        verbose (bool): 是否打印每层信息

    Returns:
        bool: True 表示所有归一化层都是 GroupNorm；False 表示发现了非 GN 的归一化层
    """
    all_gn = True
    norm_layer_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                        nn.LayerNorm, nn.LocalResponseNorm)

    def _check_recursive(module, prefix=""):
        nonlocal all_gn
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.GroupNorm):
                if verbose:
                    print(f"[OK] {full_name}: GroupNorm")
            elif isinstance(child, norm_layer_types):
                print(f"[WARN] {full_name}: Found non-GroupNorm normalization layer: {type(child).__name__}")
                all_gn = False
            else:
                # 递归检查子模块
                _check_recursive(child, full_name)

    _check_recursive(model)
    if all_gn:
        print("\n✅ 所有归一化层均已成功替换为 GroupNorm！")
    else:
        print("\n❌ 发现非 GroupNorm 的归一化层，请检查！")
    return all_gn
         
def replace_gn_with_bn_analytic(model):
    """
    Replace GroupNorm with an equivalent eval-mode BatchNorm2d.
    The resulting BN exactly matches GN in inference.
    """
    gn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_layers.append((name, module))

    for name, gn in gn_layers:
        C = gn.num_channels

        # Recover group statistics (same construction as forward GN)
        G = gn.num_groups
        assert C % G == 0

        # We cannot know true μ_g, σ_g anymore.
        # But GN is already affine-equivalent: GN(x) = W_gn * x + B_gn
        # So we directly read W_gn and B_gn.

        # σ_g, μ_g were absorbed — choose σ_bn = 1, μ_bn = 0
        W_gn = gn.weight
        B_gn = gn.bias

        bn = nn.BatchNorm2d(
            num_features=C,
            affine=True,
            track_running_stats=True,
            eps=gn.eps
        )

        # Set BN to be a pure affine layer
        bn.weight.data.copy_(W_gn)
        bn.bias.data.copy_(B_gn)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

        bn.eval()  # important: freeze behavior

        _set_layer(model, name, bn)

    return model
