# 重建 standardized 目录计划

## 目标
重建 `/data/MRI_data/standardized/`，只收录 MRI_json_reviewed 中有标注的 **502 个患者**（343 positive + 159 negative）。

## 数据来源

### 标注来源（决定收录哪些患者）
| 来源 | 数量 | 说明 |
|------|------|------|
| `MRI_json_reviewed/positive/` | 208 | 含 6 个特殊字符目录名 |
| `MRI_json_reviewed/positive_1210_MRI/` | 135 | 新增正例（MR 子目录） |
| `MRI_json_reviewed/negative/` | 159 | 含 3 个特殊字符目录名（含 N_11_ZX（delete），用户要求纳入） |

### 图像来源
| 患者组 | 图像路径 |
|--------|----------|
| 208 original positive | `MRI_img/Positive LNM/{patient_id}/` |
| 135 new positive 中 61 个 | `MRI_img/Positive LNM/{patient_id}/`（已在主目录） |
| 135 new positive 中 74 个 | `MRI_supplement/MRI_img/MRI_img/{patient_id}/`（仅在 supplement） |
| 159 negative | `MRI_img/Negative/{patient_id}/` |

### 特殊字符标注目录的 patient_id 映射
标注目录名中含括号备注，图像目录使用干净名字，需要做映射：
- `Y_135_YL(一大片）` → `Y_135_YL`
- `Y_210_CYC(淋巴结融合）` → `Y_210_CYC`
- `Y_271_HXQ（淋巴结融合）` → `Y_271_HXQ`
- `Y_522_ZBY（从这勾）` → `Y_522_ZBY`
- `Y_79_TY（一大片）` → `Y_79_TY`
- `Y_865_HSL（癌肿突出表面）` → `Y_865_HSL`
- `N_11_ZX（delete）` → `N_11_ZX`
- `N_2021_LLH（肿物突出体表）` → `N_2021_LLH`
- `N_353_zly（增强混有平扫）` → `N_353_zly`

### 子目录命名差异
- 主图像目录: `MRI/` 或 `mri/`
- Supplement 图像: 全部是 `MR/`
- positive_1210_MRI 标注: `MR/`(74), `mri/`(32), `MRI/`(29)

### 序列名映射
在原有 SEQUENCE_MAP 基础上需新增（来自 supplement 和 positive_1210_MRI）：
- DCE: `PH1`→DCE1, `PH2`→DCE2, ..., `PH7`→DCE7（PH = phase）
- `PH8`, `D8`, `D9`, `D10` → 超出 7 期，忽略
- `T2-WATER`, `T2-FAT`, `T2_WATER`, `T2_FAT` → 非标准 T2，忽略
- `T1-FSE` → 忽略（非标准 T1）
- `MUSE-B800`, `FOCUS-MUSE-B800` 等 → 忽略（特殊 DWI 序列）
- `VIBRANT-*`, `DYN-VIBRANT+C` 等 → 忽略（增强方法特殊序列）

## 实现方案

修改 `scripts/standardize_data.py`，核心改动：

### 1. 改为"以标注为主"的遍历逻辑
原脚本以图像目录为驱动，新脚本改为**以标注目录为驱动**：
- 遍历所有 3 个标注来源目录
- 对每个有标注的患者，查找对应的图像目录
- 只有找到标注的患者才会被收录

### 2. 新增图像源查找逻辑
```python
def find_image_dir(patient_id, label):
    # 1. 主图像目录
    # 2. Supplement 图像目录（仅 positive）
```

### 3. 扩展 find_mri_subdir
增加 `MR` 支持（supplement 和 positive_1210_MRI 使用 `MR/` 而非 `MRI/`）。

### 4. 扩展 SEQUENCE_MAP
新增 `PH1`-`PH7` 映射到 DCE1-DCE7。

### 5. 移除 should_skip_annotation_dir
不再跳过特殊字符目录，改用 patient_id 清洗函数提取干净的 patient_id。

### 6. 保留 features/ 目录
重建时保留 `standardized/features/BiomedCLIP/`，只清理 `images/`、`annotations/`、`dataset.csv`、`standardize_report.json`。

## 输出结构
```
standardized/
├── images/{positive,negative}/{patient_id}/{T1,T2,...,DCE7}/  (symlinks)
├── annotations/{positive,negative}/{patient_id}/{T1,T2,...,DCE7}/  (symlinks to .json)
├── dataset.csv
├── features/BiomedCLIP/  (保留不动)
└── standardize_report.json
```

CSV 格式与原来一致，但只包含 502 行（全部 has_annotation=1）。

## 预期结果
- 343 positive + 159 negative = **502** 个患者
- 全部 has_annotation=1
- 全部有图像和标注的 symlink
