# semseg-pt-v3m2-joint-s3dis-scannet-buildingnet

## 实验概述

基于 **PT-v3m2** 骨干网络，联合训练 S3DIS 和 BuildingNet 室内外点云语义分割任务，统一映射为 19 类语义标签。训练 400 epochs，核心评估指标为 **core_mIoU**（Core 10 类平均 IoU）。

| 项目 | 内容 |
|------|------|
| 骨干网络 | PT-v3m2 (Point Transformer v3 mid) |
| 编码器深度 | (3, 3, 3, 12, 3) |
| 编码器通道 | (48, 96, 192, 384, 512) |
| 训练 epochs | 400 |
| 优化器 | AdamW, lr=0.003, weight_decay=0.05 |
| 学习率调度 | OneCycleLR (cos annealing, pct_start=0.05) |
| Batch Size | 4 |
| AMP | enabled (float16) |
| 梯度裁剪 | max_norm=1.0 |
| Drop Path | 0.3 |
| 训练耗时 | ~3 天（2026-04-14 ~ 2026-04-17） |

## 数据集

| 数据集 | 划分 | 预处理 voxel size | 原始类数 | 统一映射后 |
|--------|------|------------------|---------|-----------|
| **S3DIS** | train: Area_1,2,3,4,6  /  val: Area_5 | 0.02 m | 13 | 19 类 |
| **BuildingNet** | train / val | 0.05 m | 30 | 19 类 |

### 统一类别体系（19 类）

| ID | 类别名 | 说明 | Core 类 |
|----|--------|------|---------|
| 0 | wall | 墙壁 | ✓ |
| 1 | floor_ground | 地板/地面 | ✓ |
| 2 | ceiling | 天花板 | ✓ |
| 3 | roof | 屋顶 | ✓ |
| 4 | beam | 梁 | ✓ |
| 5 | column | 柱子 | ✓ |
| 6 | window | 窗户 | ✓ |
| 7 | door_entrance | 门/入口 | ✓ |
| 8 | stairs | 楼梯 | ✓ |
| 9 | railing_fence | 栏杆/围栏 | ✓ |
| 10 | balcony_corridor_canopy | 阳台/走廊/雨棚 | |
| 11 | molding_parapet_buttress | 装饰线脚/女儿墙/扶壁 | |
| 12 | tower_chimney_dome | 塔楼/烟囱/穹顶 | |
| 13 | furniture_object | 家具/杂物 | |
| 14 | vegetation_vehicle | 植被/车辆 | |
| 15 | garage | 车库 | |
| 16 | roof_detail | 屋顶细部 | |
| 17 | pool | 泳池 | |
| 18 | other | 其他 | |

Core 类（共 10 类）: wall, floor_ground, ceiling, roof, beam, column, window, door_entrance, stairs, railing_fence

## 训练策略

### 损失函数（3 项联合监督）

| 损失函数 | loss_weight | 说明 |
|---------|------------|------|
| CrossEntropyLoss | 1.0 | 类别加权 CE，ignore_index=-1 |
| LovaszLoss | 1.0 | multiclass 模式，直接优化 IoU |
| FocalLoss | 0.5 | gamma=2.0，缓解类别不平衡 |

### 数据增强

- **几何变换**: RandomDropout (p=0.2), RandomRotate (绕 x/y/z 轴 ±1°), RandomScale [0.9, 1.1], RandomFlip (p=0.5), RandomJitter (σ=0.005)
- **颜色增强**: ChromaticAutoContrast (p=0.2), ChromaticTranslation (p=0.95), ChromaticJitter (p=0.95, std=0.05)
- **点云采样**: SphereCrop (sample_rate=0.6, point_max=102400)
- **特征**: NormalizeCoord, NormalizeColor, NormalizeNormal

### 类别加权

为缓解室内外点云类别分布不均的问题，对稀有类别（如 molding、pool、railing）赋予较高权重，对大类（如 floor、ceiling）权重为 1.0：

```
[1.2, 1.0, 1.2, 1.5, 4.0, 4.0, 2.5, 1.8, 2.5, 3.5,
 1.0, 1.0, 1.5, 0.8, 0.8, 1.0, 1.2, 1.0, 0.3]
```

## 最佳结果

**最佳 checkpoint: epoch 359** (core_mIoU = 0.4963)

### 验证集总体指标

| 指标 | 数值 |
|------|------|
| **mIoU** (19 类平均) | 0.4026 |
| **core_mIoU** (Core 10 类平均) | **0.4963** |
| **mAcc** (类别平均准确率) | 0.5278 |
| **allAcc** (点级准确率) | 0.8159 |

### 各类别 IoU / Acc（epoch 359）

| 类别 | IoU | Acc |
|------|-----|-----|
| wall | 0.7046 | 0.8211 |
| floor_ground | 0.9480 | 0.9552 |
| ceiling | 0.9148 | 0.9626 |
| roof | 0.1982 | 0.3939 |
| beam | 0.3228 | 0.7695 |
| column | 0.3929 | 0.4586 |
| window | 0.4778 | 0.6945 |
| door_entrance | 0.7277 | 0.8292 |
| stairs | 0.1357 | 0.2091 |
| railing_fence | 0.1399 | 0.3167 |
| balcony_corridor_canopy | 0.1467 | 0.1797 |
| molding_parapet_buttress | 0.0499 | 0.0583 |
| tower_chimney_dome | 0.1812 | 0.2497 |
| furniture_object | 0.8001 | 0.8820 |
| vegetation_vehicle | 0.3853 | 0.5158 |
| garage | 0.1547 | 0.2703 |
| roof_detail | 0.1986 | 0.5311 |
| pool | 0.1880 | 0.2229 |
| other | 0.5833 | 0.7080 |

## 训练曲线分析

训练曲线（`training_curves.png`）显示：

- **快速收敛阶段**（epoch 1-150）: core_mIoU 从 0.31 快速上升至 0.42，模型快速学习核心几何特征
- **稳定提升阶段**（epoch 150-370）: core_mIoU 持续攀升至 0.4963，曲线稳步上升
- **收敛震荡阶段**（epoch 370-400）: core_mIoU 在 0.49 附近小幅振荡，未出现明显过拟合，allAcc 维持在 0.80-0.82

**结论**: 模型在 epoch 359 附近达到最佳，后续 40 个 epoch 无明显增益，建议后续训练采用 **early stopping**（patience=20~30）以节省训练时间。

## 各 Core 类性能分析

| 类别 | IoU | 性能评价 |
|------|-----|---------|
| floor_ground | 0.9480 | 极高，S3DIS 室内地面特征明显 |
| ceiling | 0.9148 | 极高，天花板几何特征稳定 |
| wall | 0.7046 | 高，平面结构易识别 |
| door_entrance | 0.7277 | 高，入口区域特征可区分 |
| furniture_object | 0.8001 | 高，家具与墙面有颜色/形状差异 |
| other | 0.5833 | 中，残差类别覆盖范围广 |
| column | 0.3929 | 中，柱子在 BuildingNet 中占比少 |
| vegetation_vehicle | 0.3853 | 中，室外类别在 S3DIS 几乎不存在 |
| window | 0.4778 | 中，玻璃面与墙面易混淆 |
| beam | 0.3228 | 低，梁结构较细，容易漏检 |
| roof | 0.1982 | 低，BuildingNet 屋顶多样性高 |
| stairs | 0.1357 | 极低，楼梯结构复杂，样本少 |
| railing_fence | 0.1399 | 极低，细长结构难以精确分割 |

**主要瓶颈**: stairs（楼梯）和 railing_fence（栏杆）IoU 极低，主要原因是：
1. 细长几何结构在 voxel 化后信息损失严重
2. BuildingNet 中楼梯/栏杆样本量少，类别不平衡
3. Core 类中的 10/11 类（balcony/molding）未被计入 core_mIoU，但实际表现更差

## 文件结构

```
exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet/
├── model/
│   ├── model_best.pth        # 最佳模型（epoch 359, core_mIoU=0.4963）
│   └── model_last.pth        # 最终模型（epoch 400）
├── training_curves.png       # 训练曲线（4指标合一）
├── training_curves_core.png   # core_mIoU 单独曲线
├── training_curves_miou.png  # mIoU 单独曲线
├── training_curves_class_heatmap.png  # 19类 IoU 热力图
├── plot_training_curves.py  # 曲线绑图脚本
├── config.py                 # 训练配置快照
└── train.log                  # 完整训练日志
```

## 运行方式

```bash
# 训练
cd /path/to/Pointcept
python tools/train.py configs/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet.py

# 可视化分割效果
cd /home/yang/PointCloud_Datasets
python /mnt/c/Yang/Pointcept-main/Pointcept-main/exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet/code/viz_semseg_comparison.py
```
