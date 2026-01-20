# OCT 横向分辨率计量系统 - 项目概述

**版本**: v1.0  
**更新日期**: 2026-01-20

---

## 1. 项目目标

开发一套自动化的 **OCT 横向分辨率 (Lateral Resolution)** 计量验证系统，依据校准规范，使用标准微球体模测量系统的点扩散函数 (PSF) 半高全宽 (FWHM)，输出统计指标供计量报告使用。

### 1.1 核心功能

| 功能 | 说明 |
|------|------|
| **全自动检测** | DoG 多尺度斑点检测，识别数百个候选微球 |
| **多级筛选** | SNR、形态、强度、边界等多维度质量控制 |
| **亚像素测量** | 2D 高斯拟合实现 0.1px 级定位精度 |
| **统计分析** | 输出均值/标准差/极值，警告样本不足 |
| **交互式界面** | 实时参数调整，ROI 选择，结果叠加显示 |

### 1.2 典型应用场景

- 眼科 OCT 设备出厂检测
- 计量检测机构性能验证
- 科研实验室自建 OCT 系统评估

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    GUI 表现层                        │
│  app/main.py (PySide6 深色主题界面)                  │
├─────────────────────────────────────────────────────┤
│                    业务服务层                        │
│  app/services/analysis_service.py                   │
├─────────────────────────────────────────────────────┤
│                    核心算法层                        │
│  core/analysis.py   - 主流水线                      │
│  core/detection.py  - DoG 检测                      │
│  core/fwhm.py       - FWHM 计算                     │
│  core/profile.py    - 横向 Profile 提取              │
│  core/models.py     - 数据模型                       │
├─────────────────────────────────────────────────────┤
│                    配置与数据层                      │
│  config/default_config.json                         │
│  data_samples/                                      │
└─────────────────────────────────────────────────────┘
```

---

## 3. 快速开始

### 3.1 环境要求

- Python 3.10+
- 依赖: PySide6, numpy, scipy, scikit-image, matplotlib

### 3.2 安装与运行

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python app/main.py
```

### 3.3 基本使用流程

1. **加载图像**: 点击 "Open Image" 选择 B-scan 图像
2. **设置像素标尺**: 在参数面板输入 µm/px 值
3. **选择 ROI** (可选): 使用 "Select ROI" 工具框选分析区域
4. **运行分析**: 点击 "Run Analysis"
5. **查看结果**: 统计面板显示分辨率，表格显示每个微球详情

---

## 4. 文件结构

```
OCT_LatRes/
├── app/                    # GUI 应用
│   ├── main.py            # 主窗口
│   └── services/          # 业务服务
├── core/                   # 核心算法
│   ├── analysis.py        # 主分析流水线
│   ├── detection.py       # DoG 检测
│   ├── fwhm.py            # FWHM 计算
│   ├── models.py          # 数据模型
│   └── profile.py         # Profile 提取
├── config/                 # 默认配置
├── docs/                   # 项目文档
├── tests/                  # 测试用例
├── scripts/                # 辅助脚本
└── archive/                # 归档的实验代码
```

---

## 5. 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v1.0 | 2026-01-20 | 首个稳定里程碑；代码架构规范化 |
| v0.1 | 2026-01-15 | 初始原型；DoG + 离散 FWHM |
