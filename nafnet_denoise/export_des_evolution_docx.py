"""Export DES evolution + failures report to Word (.docx) for Google Docs import."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

OUT = Path("nafnet_denoise/docs/DES_evolution_P150_vs_BM3D.docx")

BM3D = 0.8882

LADDER = [
    ("—", "4f 单路 adaptive", 0.851, "低于 BM3D", "神经网络", "多数 clip 低于 BM3D"),
    ("P0", "FE Wiener 门控", 0.905, "+0.017", "融合", "首次均值超 BM3D"),
    ("P4", "SOTA↔Edge 双路 blend", 0.906, "+0.018", "融合", "T=8 harden=16"),
    ("P5", "低 fps 平坦区 bilateral", 0.910, "+0.022", "后处理", "s=0.65"),
    ("P7", "多 band bilat 调度", 0.918, "+0.030", "后处理", "Gen2 bake"),
    ("P10", "几何 TTA id+lr", 0.923, "+0.035", "TTA", "2 aug"),
    ("P13", "D4 r90 edge FT", 0.925, "+0.037", "训练", "edge 路微调"),
    ("P22", "fps 分段 unsharp+bilat", 0.929, "+0.041", "后处理", "正式超 BM3D 对比点"),
    ("P40", "fps band 1.67/5", 0.929, "+0.041", "后处理", "微调"),
    ("P68", "MAD 噪声门控", 0.944, "+0.056", "后处理", "★ 大跃升"),
    ("P98", "Anscombe flat shrink", 0.947, "+0.059", "小波", "VST 域"),
    ("P103", "noise-gated Anscombe", 0.951, "+0.062", "小波", "ng 门控"),
    ("P114", "CNE + cycle-spin", 0.953, "+0.065", "小波", "YOND lite"),
    ("P120", "递归 cycle-spin n=2", 0.956, "+0.067", "小波", "Coifman-Donoho"),
    ("P126", "EUI 逆变换 n=3", 0.957, "+0.068", "小波", "Makitalo-Foi"),
    ("P134", "SURE-k 选阈值", 0.957, "+0.069", "小波", "k∈{0.8,1.0,1.2}"),
    ("P142", "BlockJS + SURE", 0.958, "+0.070", "小波", "James-Stein"),
    ("P150", "BiShrink + SURE n=5", 0.959, "+0.071", "小波", "★ 当前 SOTA"),
]

FAILURES = [
    ("起点", "4f", "单路 adaptive NAFNet", "~0.851", "低于BM3D", "11 clip 多数低于 BM3D"),
    ("P4", "P4d", "单网 blend distill", "0.904", "拒绝", "落后双路 0.906"),
    ("P4", "P4b", "residual mix", "≤0.905", "拒绝", "mean 降"),
    ("P4", "P4c", "f05 平坦特化", "—", "拒绝", "ng 无提升"),
    ("P5", "P5a", "Laplacian multiband", "0.906", "拒绝", "无差异"),
    ("P5", "P5b", "naive halfscale unsharp", "~0.916", "DES gaming", "edge_safe 失败"),
    ("P5", "P5c", "flat-mask BM3D 混合", "0.893", "低于BM3D", "mean 低于 deploy"),
    ("P5", "P5d", "Noise2N 时域 FT", "0.887", "低于BM3D", "跌破 BM3D"),
    ("P5", "P5e", "Feature KD", "—", "拒绝", "distill 失败"),
    ("P7-9", "loop100", "300 配方 auto-loop", "0.918*", "DES gaming", "f05_ef 下限"),
    ("P7-9", "P9", "DualHead 短训 FT", "~0.922", "拒绝", "plateau 无 bake"),
    ("P13-16", "P16", "D4 频域 split fuse", "0.924", "拒绝", "未超 P13"),
    ("P22-40", "P47", "guided filter", "≤0.929", "拒绝", "调度饱和"),
    ("P68-114", "P109", "8×8 WHT", "—", "拒绝", "字典不适用"),
    ("P68-114", "P112", "spatial SNR Anscombe", "—", "拒绝", "无 bake"),
    ("P68-114", "P113", "Haar on P114", "≤0.953", "拒绝", "无增量"),
    ("P115-131", "clone", "schedule/Anscombe 克隆", "≤0.953", "饱和", "5 连 no-bake"),
    ("P132-141", "P132-135", "σz-gate/EM-VST/DualDn", "—", "拒绝", "无 bake"),
    ("P132-141", "P140-141", "garrote/firm shrink", "≤0.958", "拒绝", "BiShrink 更优"),
    ("P142-155", "P148-149", "NeighShrink", "≤0.958", "拒绝", "无 bake"),
    ("P142-155", "P151", "SURE-LET mix", "≤0.9589", "拒绝", "无 bake"),
    ("P142-155", "P152-155", "参数 clone", "≤0.9589", "饱和", "early-stop"),
    ("P156-159", "P156-158", "NeighLevel/Bayes/BiNeigh", "≤0.9589", "拒绝", "未超 P150"),
    ("P156-159", "P159", "SURE 选 shrink 模式", "—", "bug", "use_eui 重复传参"),
    ("全局", "—", "Fourier/EnsIR/NLMeans/TTA均值", "—", "拒绝", "dead-end"),
]


def add_table(doc: Document, headers: list[str], rows: list[tuple]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            table.rows[ri + 1].cells[ci].text = str(val)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()

    title = doc.add_heading("Mono10 RAW 降噪 DES 演进报告", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "从泊松分布验证与 PTC 标定起步，经 DES 基准体系建立，"
        "早期单路 adaptive 效果差于 BM3D（~0.851），"
        "到 P150（BiShrink + SURE-k）领先 BM3D +0.071。"
        "本文档含标定/训练素材、成功 bake 路径、失败/拒绝尝试、P150 流程与 BM3D 对比。"
    )
    doc.add_paragraph(
        "生成日期：2026-08-23 · 数据源：common.py / train_distill.py / "
        "burst manifests / compare_p150_vs_bm3d / cycle logs"
    )

    doc.add_heading("一、核心指标", level=1)
    add_table(
        doc,
        ["指标", "数值"],
        [
            ("VST+BM3D 基线 mean DES", "0.8882"),
            ("P150 最终 mean DES", "0.9589"),
            ("领先 BM3D", "+0.0707"),
            ("edge_safe", "true"),
            ("最优 recipe", "bi_0.8_1_1.2_n5_d0.7"),
            ("SOTA checkpoint", "checkpoints_4f_split_edge_fid/best.pt"),
            ("Edge checkpoint", "checkpoints_4f_p12_deploy_edge/best.pt"),
        ],
    )

    doc.add_heading("二、标定与基准体系（Phase 0–3）", level=1)

    doc.add_heading("Phase 0 · 泊松分布（Poisson-Gaussian）噪声验证", level=2)
    doc.add_paragraph(
        "项目起点：验证 Mono10 传感器噪声是否服从 Poisson-Gaussian（散粒噪声 + 读出噪声）模型。"
        "理论形式为 variance_dn² = slope × mean_dn + intercept，"
        "其中 slope > 0 且随 mean_dn 近似线性，即支持泊松散粒噪声假设；"
        "intercept 为读出噪声方差项。相关脚本与报告位于 泊松分布/ 目录。"
    )

    doc.add_heading("0.1 平坦场时域 PTC 验证", level=3)
    doc.add_paragraph(
        "脚本：泊松分布/validate_ptc_mono10.py。"
        "对 8 条不同 fps（f5–f20）的平坦场 Mono10 视频，"
        "在中心 ROI（40% 宽高）逐像素计算时域样本方差（非空间方差，排除固定图案 PRNU），"
        "并对每帧 ROI 均值做 flicker 校正后再拟合 PTC。"
    )
    add_table(
        doc,
        ["项", "结果"],
        [
            ("分析序列", "8 条平坦场 RAW（f5–f20，mean_dn 303–955）"),
            ("拟合", "variance_dn² = 0.0955 × mean_dn + 4.436"),
            ("R²", "0.728"),
            ("读出噪声估计", "√intercept ≈ 2.11 DN"),
            ("结论", "正斜率支持 Poisson-Gaussian；部分 clip 因灯光闪烁 R² 偏低"),
            ("输出", "泊松分布/ptc_validation/ptc_report.txt + ptc_summary.csv"),
        ],
    )

    doc.add_heading("0.2 灰阶图高精度 PTC 标定", level=3)
    doc.add_paragraph(
        "脚本：泊松分布/calibrate_ptc_grayscale_video.py。"
        "素材：D:\\denoise\\素材\\实验素材 泊松\\ 下静态灰阶图视频。"
        "逐像素时域方差 → 按 temporal mean DN 分 bin → 鲁棒线性拟合；"
        "帧级仿射亮度归一化抑制灯光闪烁，不将空间 PRNU 混入时域方差。"
        "两次独立录制取均值，写入 nafnet_denoise/common.py："
    )
    add_table(
        doc,
        ["录制", "slope", "intercept", "R²", "读出噪声 (DN)"],
        [
            ("Video_43338452 f20", "0.106221", "2.033", "0.99953", "1.426"),
            ("Video_43519648 f20", "0.106149", "1.969", "0.99956", "1.403"),
            ("均值 → common.py", "0.106185", "2.001456", "—", "~1.41"),
        ],
    )
    doc.add_paragraph(
        "报告路径：泊松分布/ptc_grayscale_videos/<video>/ptc_report.txt。"
        "usable flat pixels > 1.6M，57+ gray bins，frame scale CV < 0.4%。"
    )

    doc.add_heading("0.3 PTC 标定优势验证", level=3)
    doc.add_paragraph(
        "脚本：泊松分布/compare_ptc_advantage.py。"
        "在验证视频上对比 blind BM3D（帧差盲估 σ）与 PTC 引导的 VST+BM3D："
        "后者使用已验证的 PTC 参数做 generalized-Anscombe 变换，"
        "降噪更贴合真实 signal-dependent 噪声，为后续 BM3D teacher 与合成训练噪声提供依据。"
    )

    doc.add_heading("Phase 0 · PTC 标定与管线参数", level=2)
    doc.add_paragraph(
        "Poisson-Gaussian 验证通过后，将 PTC 常数固化到全管线（common.py），"
        "并继续完成 VST 域与后处理标定："
    )
    add_table(
        doc,
        ["常量", "数值", "含义"],
        [
            ("PTC_SLOPE", "0.106185", "variance_dn² = slope × mean_dn + intercept"),
            ("PTC_INTERCEPT", "2.001456", "读出噪声截距 (DN²)"),
            ("DEFAULT_BLACK_LEVEL_DN", "60", "默认黑电平"),
            ("RAW_MAX", "1023", "Mono10 满量程"),
        ],
    )
    p0_steps = [
        "Mono10 解码 → float32 DN；曝光 intercept 与 fps/exposure_ms 解析",
        "VST 正/逆变换：将 Poisson-Gaussian 噪声近似方差稳定化",
        "calibrate_postmerge_noise.py：Wiener 合并后平坦区残差 σ 仿射拟合 → postmerge_noise.json",
        "calibrate_hard_bm3d_sigma.py：f0.5/f10 硬场景逐 clip 网格搜索 BM3D σ_psd",
    ]
    for s in p0_steps:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("Phase 1 · DES 评测基准", level=2)
    doc.add_paragraph(
        "DES（Denoise-Edge Score）定义于 train_distill.py，用于所有 P-cycle bake 决策："
    )
    add_table(
        doc,
        ["分量", "公式", "权重"],
        [
            ("noise_gain", "clip(1 − σ_flat / σ_input, 0, 1)", "0.6"),
            ("edge_fidelity", "clip(1 − |edge_retention − 1|, 0, 1)", "0.4"),
            ("DES", "0.6 × noise_gain + 0.4 × edge_fidelity", "—"),
        ],
    )
    doc.add_paragraph(
        "edge_safe 约束（run_des_loop100.py）：f10 edge_fidelity ≥ 0.965，"
        "f0.5 ≥ 0.870，f1 ≥ 0.960。不满足则配方拒绝（DES gaming）。"
    )
    p1_items = [
        "Holdout：11 条 Mono10 RAW（D:\\denoise\\val_raw ≡ D:\\denoise\\素材\\降噪素材），fps 0.5–250",
        "参考：aligned temporal trimmed-mean + reliable_reference_mask（benchmark_multiframe.py）",
        "P-cycle 推理缓存：nafnet_denoise/cache_dual_holdout/（SOTA/edge/input/ref/mask + TTA tag）",
        "锚点 clip：f0.5（53940814）、f1（53741354）、f10（74824541）用于 edge_safe 门禁",
    ]
    for s in p1_items:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("Phase 2 · BM3D 基线", level=2)
    doc.add_paragraph(
        "build_bm3d_teacher.py 在 VST 域对每条 burst 生成离线 BM3D teacher；"
        "compare 脚本在 11 clip holdout 上测得 VST+BM3D mean DES = 0.8882，"
        "作为全部后续 P-cycle 的对照基线。"
    )

    doc.add_heading("Phase 3 · 神经网络训练阶梯", level=2)
    doc.add_paragraph(
        "训练采用「合成 + 真机 burst + BM3D/temporal 分区 teacher」蒸馏路线（train_distill.py）："
    )
    train_ladder = [
        ("1", "PixelShift200 合成预训", "checkpoints_4f_ptc_black60/best.pt", "200 张 clean GT + PTC 合成噪声"),
        ("2", "BM3D teacher 蒸馏", "checkpoints_4f_ptc_black60_bm3d_distill/", "合成 30% + 真机 burst 70%"),
        ("3", "Edge teacher + Wiener FE", "checkpoints_4f_dual_head_nonlocal/", "flat=BM3D / edge=temporal 分区"),
        ("4", "DES 分区微调", "checkpoints_4f_des_partition/", "run_des_improve.py Phase A"),
        ("5", "降噪素材 FT（可选）", "checkpoints_4f_des_material_ft/", "9/11 clip，排除 f10/f250"),
        ("6", "Split edge fidelity FT", "checkpoints_4f_split_edge_fid/best.pt", "★ SOTA 部署路"),
        ("7", "Edge 部署路 FT", "checkpoints_4f_p12_deploy_edge/best.pt", "★ 高 fps 双路融合 edge 臂"),
    ]
    add_table(doc, ["阶段", "内容", "Checkpoint", "说明"], train_ladder)
    doc.add_paragraph(
        "早期 4f 单路 adaptive 在 holdout 上 mean DES ≈ 0.851，低于 BM3D 0.8882；"
        "P4 引入 SOTA↔Edge 双路融合后才稳定超过 BM3D，后续 P-cycle 均在 deploy 融合层迭代。"
    )

    doc.add_heading("三、训练素材", level=1)

    doc.add_heading("3.1 合成数据（PixelShift200）", level=2)
    add_table(
        doc,
        ["项", "详情"],
        [
            ("原始路径", r"D:\denoise\素材\PixelShift200_train"),
            ("格式", "pixelshift_*.mat，ps4k (H,W,4) uint16 → mono luminance"),
            ("缓存", "nafnet_denoise/cache/manifest.json"),
            ("条目数", "200"),
            ("用途", "PixelShiftPatchDataset：随机 patch + PTC Poisson-Gaussian 合成噪声"),
            ("构建脚本", "python -m nafnet_denoise.build_dataset"),
        ],
    )

    doc.add_heading("3.2 真机 Burst 训练集", level=2)
    add_table(
        doc,
        ["项", "详情"],
        [
            ("原始路径", r"D:\denoise\素材\训练素材_old"),
            ("子目录", "北面电脑 / 小房间 / 白板区域 / 被子等 / 不同曝光的素材 等"),
            ("排除规则", "路径含「不参加训练」的目录自动排除（build_burst_dataset.py）"),
            ("RAW 总数", "33 条（排除验证集后 31 条入训）"),
            ("缓存", "nafnet_denoise/burst_cache_train/manifest.json（31 entries）"),
            ("对齐参考", "时域 trimmed-mean clean + reliability mask（64 ref frames）"),
            ("Teacher", "build_bm3d_teacher.py → __bm3d.npy（VST+BM3D 离线）"),
            ("Dataset", "RealBurstDataset（burst_dataset.py，16/4 帧窗口 + D4 增广）"),
        ],
    )

    doc.add_heading("3.3 Holdout / 评测素材（不参与 DES bake 训练）", level=2)
    add_table(
        doc,
        ["项", "详情"],
        [
            ("路径", r"D:\denoise\val_raw ≡ D:\denoise\素材\降噪素材"),
            ("Clip 数", "11"),
            ("fps 范围", "0.5, 1, 1.25, 1.67, 2.5, 5, 10, 83, 125, 167, 250"),
            ("用途", "DES 评测、P-cycle bake、P150 vs BM3D 对比、素材批推理"),
            ("缓存", "nafnet_denoise/burst_cache_denoise_material/manifest.json（11 entries）"),
            ("Phase-B FT 子集", "manifest_train.json（9 entries，排除 _f10.raw 与 _f250.raw）"),
            ("硬场景标定", "manifest_hard_flat.json（f0.5 + f10，BM3D σ 网格）"),
        ],
    )
    doc.add_paragraph("11 clip 明细：")
    holdout_clips = [
        ("f0.5", "Video_20260719153940814_w1920_h1200_pMono10_f0.5.raw", "edge_safe 锚点（低 fps 硬场景）"),
        ("f1", "Video_20260719153741354_w1920_h1200_pMono10_f1.raw", "edge_safe 锚点"),
        ("f1.25", "Video_20260725174729355_w1920_h1200_pMono10_f1.25.raw", "—"),
        ("f1.67", "Video_20260725174622596_w1920_h1200_pMono10_f1.67.raw", "—"),
        ("f2.5", "Video_20260725174552676_w1920_h1200_pMono10_f2.5.raw", "—"),
        ("f5", "Video_20260725174703305_w1920_h1200_pMono10_f5.raw", "—"),
        ("f10", "Video_20260725174824541_w1920_h1200_pMono10_f10.raw", "edge_safe 锚点；Phase-B FT 排除"),
        ("f83", "Video_20260723180937926_w1920_h1200_pMono10_f83.raw", "—"),
        ("f125", "Video_20260723180854451_w1920_h1200_pMono10_f125.raw", "—"),
        ("f167", "Video_20260723181000246_w1920_h1200_pMono10_f167.raw", "—"),
        ("f250", "Video_20260723181024133_w1920_h1200_pMono10_f250.raw", "Phase-B FT 排除"),
    ]
    add_table(doc, ["fps", "文件名", "备注"], holdout_clips)

    doc.add_heading("3.4 仅验证集（训练时不用）", level=2)
    doc.add_paragraph(
        r"D:\denoise\素材\训练素材_old\不参加训练，用来验证训练模型效果"
        " — validate.py / validate_burst.py 默认 validation 目录，"
        "build_burst_dataset 默认 exclude。"
    )

    doc.add_heading("3.5 训练/推理数据流（简图）", level=2)
    flow = doc.add_paragraph()
    flow.add_run(
        "PixelShift200 (200) ──┐\n"
        "                      ├── train_distill ──→ SOTA ckpt (split_edge_fid)\n"
        "真机 burst (31) ──────┤         │\n"
        "  + BM3D teacher      │         └──→ Edge ckpt (p12_deploy_edge)\n"
        "                      │\n"
        "降噪素材 holdout (11) ─┴── DES 评测 / P-cycle bake（不用于主训练）\n"
        "                      └── Phase-B FT 可选（9/11，排除 f10/f250）"
    ).font.name = "Consolas"
    flow.runs[0].font.size = Pt(9)

    doc.add_heading("四、演进曲线（ASCII）", level=1)
    curve = doc.add_paragraph()
    curve.add_run(
        "Phase 0  Poisson-Gaussian 验证 (R²≈1.0) → PTC 标定 → VST\n"
        "Phase 1  DES 基准 → Phase 2 BM3D 0.888\n"
        "Phase 3  4f adaptive ~0.851（低于 BM3D）\n"
        "  ↑ P4 blend 0.906 (P4d/P5d 等失败分支见下表)\n"
        "  ↑ P5–P13  0.910–0.925 (TTA + edge FT)\n"
        "  ↑ P22     0.929 ─── 平台期 ~0.929 (数百配方 no-bake) ───\n"
        "  ↑ P68     0.944 ★ MAD 门控跃升\n"
        "  ↑ P98–142 0.947–0.958 (Anscombe/小波链)\n"
        "  ↑ P150    0.959 ★ BiShrink + SURE-k\n"
        "  × P151–159 饱和 early-stop"
    ).font.name = "Consolas"
    curve.runs[0].font.size = Pt(9)

    doc.add_heading("五、成功里程碑（烘焙 bake）", level=1)
    add_table(
        doc,
        ["Cycle", "方法", "mean DES", "vs BM3D", "类型", "说明"],
        [(c, m, f"{d:.3f}", v, t, n) for c, m, d, v, t, n in LADDER],
    )

    doc.add_heading("六、失败 / 拒绝 / 饱和 尝试", level=1)
    doc.add_paragraph(
        "共记录 25+ 类 dead-end。类型说明：低于BM3D = mean 不及 BM3D；"
        "DES gaming = mean 升但 edge_safe 失败；饱和 = P150 后连续 no-bake。"
    )
    add_table(
        doc,
        ["阶段", "Cycle", "尝试", "DES", "类型", "失败原因"],
        FAILURES,
    )

    doc.add_heading("七、失败阶段总结", level=1)
    phases = [
        ("A · 低于 BM3D", "4f 单路、P5c BM3D 混合、P5d N2N — 需双路融合而非纯网络/BM3D 混合"),
        ("B · DES gaming", "P5b unsharp、loop100 Gen3 — mean 升但 edge 约束失败"),
        ("C · P22–P68 平台", "fps/bilat/unsharp 调度饱和，guided filter 等无 bake"),
        ("D · 小波 dead-end", "WHT、Haar、garrote、NeighShrink alone — BlockJS/BiShrink 胜出"),
        ("E · P150 后饱和", "LET、NeighLevel、Bayes、clone — 5 连 no-bake stop"),
    ]
    for name, desc in phases:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{name}：").bold = True
        p.add_run(desc)

    doc.add_heading("八、P150 降噪流程（详细）", level=1)
    doc.add_paragraph(
        "入口：infer_blend_deploy.py → denoise_blend_deploy()，默认 ans_mode=bishrink（P150）。"
        "整条管线从 Mono10 RAW 到最终 DN 输出分为 6 个阶段；"
        "P150 的核心创新在阶段 5–6：MAD 噪声门控后处理 + SURE 选阈值 + "
        "递归 EUI cycle-spin + BiShrink 平坦区收缩。"
    )

    doc.add_heading("8.1 总体数据流", level=2)
    flow_p150 = doc.add_paragraph()
    flow_p150.add_run(
        "Mono10 RAW (memmap)\n"
        "  └─[1] 16帧时域 Wiener 对齐合并 → temporal\n"
        "       └─[2] FE 门控 spatial Wiener → fe_gated\n"
        "            └─[3] NAFNet SOTA (+6 TTA) → sota_dn\n"
        "                 └─ (fps>1.5) Edge NAFNet (+6 TTA) → edgekd_dn\n"
        "                      └─[4] fps 路由 + SOTA↔Edge 软融合\n"
        "                           └─[5] MAD 门控 bilateral + unsharp (P68)\n"
        "                                └─[6] SURE-k × 递归 EUI cycle-spin BiShrink (P150)\n"
        "                                     → 最终降噪 DN"
    ).font.name = "Consolas"
    flow_p150.runs[0].font.size = Pt(9)

    doc.add_heading("8.2 阶段 1：时域 Wiener 多帧合并", level=2)
    doc.add_paragraph(
        "函数：wiener_merge.merge_from_memmap()。"
        "以目标帧为中心取 16 帧窗口，逐帧亮度对齐后做时域 Wiener 合并，"
        "抑制帧间独立噪声、保留静态结构。"
    )
    add_table(
        doc,
        ["参数", "值", "说明"],
        [
            ("temporal_window", "16", "参与合并的帧数"),
            ("wiener_tile / overlap", "32 / 16", "分块 Wiener 尺寸与重叠"),
            ("c_factor", "8.0", "时域 Wiener 保守系数"),
            ("align", "True", "帧间平移对齐后再合并"),
            ("spatial_wiener", "False", "此步仅时域，空间 Wiener 在 FE 阶段"),
        ],
    )

    doc.add_heading("8.3 阶段 2：FE 门控 spatial Wiener", level=2)
    doc.add_paragraph(
        "函数：fe_schedule.gated_spatial_from_temporal()。"
        "根据 fps 与合并后残差 σ 自动选择 preset（ea_mild_gamma / baseline_spatial / ea_mid_soft），"
        "对 temporal 结果做 edge-aware 空间 Wiener，输出 fe_gated 供神经网络输入。"
        "平坦区更强去噪（flat_c_mult=1.75），边缘区更保守（edge_c_mult=0.45），"
        "dark_boost=0.35 补偿暗部。"
    )

    doc.add_heading("8.4 阶段 3：神经网络推理 + 几何 TTA", level=2)
    doc.add_paragraph(
        "函数：_spatial_on_merged() → denoise_from_dn_frames()。"
        "将 fe_gated 复制为 4 帧模型输入（4f NAFNet），"
        "关闭模型内置 wiener_front_end（已在 FE 阶段完成）。"
        "对 6 种几何变换分别推理后逆变换取均值，降低方向性伪影："
    )
    tta_items = [
        "id — 原图",
        "r90 / r180 / r270 — 旋转 90°/180°/270°",
        "r90_lr / r270_lr — 旋转 + 水平翻转",
    ]
    for s in tta_items:
        doc.add_paragraph(s, style="List Bullet")
    add_table(
        doc,
        ["网络", "Checkpoint", "用途"],
        [
            ("SOTA 路", "checkpoints_4f_split_edge_fid/best.pt", "平坦区保真，所有 fps 均运行"),
            ("Edge 路", "checkpoints_4f_p12_deploy_edge/best.pt", "仅 fps > 1.5 时运行"),
        ],
    )

    doc.add_heading("8.5 阶段 4：fps 感知双路路由", level=2)
    add_table(
        doc,
        ["条件", "行为"],
        [
            ("fps ≤ 1.5（低 fps）", "仅 SOTA 路；deploy 函数输入 (sota, sota)；route=sota_noise_bilat_unsharp_anscombe"),
            ("fps > 1.5（中高 fps）", "SOTA + Edge 双路；route=blend_noise_bilat_unsharp_anscombe"),
        ],
    )
    doc.add_paragraph(
        "高 fps 双路融合（blend_sota_edgekd）：guide = 0.5×sota + 0.5×edge，"
        "由 Sobel 梯度生成 soft edge map（temperature=8, harden=16），"
        "输出 = edge_map × edge + (1 − edge_map) × sota。"
        "边缘区域保留 Edge 网络的锐度，平坦区沿用 SOTA 的去噪强度。"
    )

    doc.add_heading("8.6 阶段 5：MAD 噪声门控后处理（P68）", level=2)
    doc.add_paragraph(
        "函数：deploy_p68_noise()。"
        "在平坦区估计 MAD 噪声 proxy n，将其线性映射到 bilateral 与 unsharp 强度："
    )
    add_table(
        doc,
        ["噪声 proxy n", "映射范围", "效果"],
        [
            ("n ∈ [0.002, 0.012]", "bilateral 0.72 → 1.0", "噪声越大，平坦区 bilateral 越强"),
            ("n ∈ [0.002, 0.012]", "unsharp 0.10 → 0.22", "噪声越大，边缘 unsharp 越强"),
            ("flat_pct", "30%", "MAD 估计仅在平坦 percentile 内采样"),
        ],
    )
    p68_steps = [
        "flat_bilateral_boost：仅在 Sobel 平坦掩膜内做 edge-preserving bilateral（harden=40）",
        "edge_unsharp：对边缘区做 amount 自适应 unsharp（sigma=1.4, harden=16）",
        "此阶段输出作为 P150 小波收缩的基底图像 base",
    ]
    for s in p68_steps:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("8.7 阶段 6：P150 核心 — SURE-k + 递归 EUI BiShrink", level=2)
    doc.add_paragraph(
        "函数：deploy_p150() → _sure_pick_k(bishrink=True)。"
        "对 k_mad ∈ {0.8, 1.0, 1.2} 各跑一遍完整递归收缩，"
        "用 SURE-lite 风险函数在平坦区选最优 k，避免手工调阈值。"
    )

    doc.add_heading("6a. SURE-k 选择", level=3)
    doc.add_paragraph(
        "对每个候选 k：运行 _recursive_cne_spin_core(ans_k_mad=k, bishrink=True)，"
        "在平坦+低 Sobel 掩膜上计算："
        "score = mean(|cand − base|²) + 2×mad₀²×keep_fraction。"
        "keep_fraction 为残差超过 k×mad₀ 的像素比例（惩罚过度收缩）。"
        "取 score 最小的 k 对应结果为最终输出。"
    )

    doc.add_heading("6b. 递归 EUI cycle-spin（5 轮）", level=3)
    doc.add_paragraph(
        "每轮迭代执行 cycle_spin_anscombe(max_shift=1)，即 4 种 (dy,dx) 平移 "
        "(0,0)(0,1)(1,0)(1,1) 的 Coifman–Donoho cycle-spin 均值，消除块效应。"
        "收缩强度 s 由 |P68输出 − SOTA| 的 MAD 经 residual_scale=0.3 映射到 [0.12, 0.4]。"
        "每轮结束后 k_mad ×= thr_decay(0.7)，阈值逐轮递减、收缩逐渐温和。"
    )

    doc.add_heading("6c. 单轮 Anscombe + BiShrink 细节", level=3)
    ans_steps = [
        "广义 Anscombe 正变换：z = √(x + 3/8)，将 Poisson-Gaussian 噪声方差稳定化",
        "高斯低通：low = GaussianBlur(z, σ=1.8)；高 pass hp = z − low",
        "平坦区噪声估计：MAD(highpass) 在 flat_pct=50% 采样 → thr = k_mad × MAD",
        "BiShrink（Sendur–Selesnick）：parent = 2× 下采样 |hp| 再上采样；"
        "r = √(hp² + parent²)；scale = max(1 − thr×√3 / r, 0)；hp' = hp × scale",
        "重建：z' = low + hp'；EUI 逆变换（Makitalo–Foi 闭式，gat_eui_inverse）→ DN 域 y",
        "平坦区混合：仅在 Sobel 平坦掩膜内以 strength=s 混合原图与 y；"
        "out = x×(1 − s×flat) + y×(s×flat)，边缘区保持 P68 结果不动",
    ]
    for i, s in enumerate(ans_steps, 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(s)

    doc.add_heading("8.8 P150 最优参数一览", level=2)
    add_table(
        doc,
        ["参数", "值", "含义"],
        [
            ("recipe", "bi_0.8_1_1.2_n5_d0.7", "deploy_p150_hook.json 烘焙名"),
            ("k_list", "0.8, 1.0, 1.2", "SURE 候选 MAD 乘子"),
            ("n_iters", "5", "递归 cycle-spin 轮数"),
            ("thr_decay", "0.7", "每轮 k_mad 衰减系数"),
            ("ans_sigma", "1.8", "Anscombe 高斯低通 σ"),
            ("residual_scale", "0.3", "残差 MAD → 收缩强度映射缩放"),
            ("ans_s_lo / ans_s_hi", "0.12 / 0.4", "收缩 strength 上下界"),
            ("max_shift", "1", "cycle-spin 平移范围（4 aug）"),
            ("use_eui", "True", "EUI 逆变换（优于代数逆）"),
            ("low_fps", "1.5", "双路融合 fps 阈值"),
            ("noise_lo / noise_hi", "0.002 / 0.012", "P68 MAD 门控范围"),
            ("TTA", "6 aug", "id/r90/r180/r270/r90_lr/r270_lr"),
        ],
    )

    doc.add_heading("8.9 为何 P150 有效", level=2)
    why_items = [
        "双路融合（P4）：SOTA 保平坦、Edge 保锐度，互补优于单路 adaptive（0.851）",
        "MAD 门控（P68）：按 clip 实际噪声自适应 bilateral/unsharp，+0.015 DES 跃升",
        "Anscombe + EUI（P98–P126）：在 VST 近似域做平坦区收缩，贴合 PTC 标定的 signal-dependent 噪声",
        "cycle-spin（P120）：4 向平移均值消除 directional artifact",
        "SURE-k（P134）：自动选 k，避免 over/under-shrink",
        "BiShrink（P150）：双变量收缩利用 parent 子带相关性，优于单变量 soft-threshold（+0.001 vs P142 BlockJS）",
    ]
    for s in why_items:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("九、P150 vs BM3D 对比（11 clip 素材）", level=1)
    add_table(
        doc,
        ["方法", "mean DES", "vs BM3D"],
        [
            ("P150 BiShrink+SURE-k", "0.9589", "+0.0707"),
            ("VST+BM3D", "0.8882", "—"),
        ],
    )
    doc.add_paragraph("数据源：nafnet_denoise/compare_p150_vs_bm3d/summary.json")

    doc.add_heading("十、关键代码入口", level=1)
    add_table(
        doc,
        ["文件", "函数/配置", "作用"],
        [
            ("infer_blend_deploy.py", "denoise_blend_deploy()", "端到端推理入口"),
            ("p8_fusion.py", "deploy_p150()", "P150 小波收缩核心"),
            ("p8_fusion.py", "_sure_pick_k()", "SURE-k 候选选择与评分"),
            ("p8_fusion.py", "_recursive_cne_spin_core()", "递归 EUI cycle-spin 循环"),
            ("p8_fusion.py", "_bishrink_hp()", "Sendur–Selesnick 双变量收缩"),
            ("deploy_p150_hook.json", "bi_0.8_1_1.2_n5_d0.7", "烘焙最优配方与 DES 分数"),
        ],
    )

    doc.add_heading("十一、导入 Google 文档步骤", level=1)
    doc.add_paragraph(
        "1. 打开 Google Drive (drive.google.com)\n"
        "2. 新建 → 文件上传 → 选择本 .docx 文件\n"
        "3. 上传完成后右键 → 打开方式 → Google 文档\n"
        "4. 可选：文件 → 另存为 Google 文档格式"
    )

    doc.save(OUT)
    print(f"Saved {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
