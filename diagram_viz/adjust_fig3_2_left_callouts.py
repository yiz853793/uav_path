from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

BASE = Path(".")
FIG32 = BASE / "paper_assets" / "fig_3_2_moead_enhanced_design.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W_NS, "r": R_NS}
ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_BOX = font(31, True)
F_SUB = font(24)
F_CALLOUT = font(25)


def qn(ns, tag):
    return f"{{{ns}}}{tag}"


def size(draw, text, fnt):
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw, text, fnt, max_w):
    lines = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            cand = line + ch
            if size(draw, cand, fnt)[0] <= max_w or not line:
                line = cand
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines


def center(draw, box, lines, fonts, fill="#1f2d3d", gap=7):
    x1, y1, x2, y2 = box
    if not isinstance(fonts, list):
        fonts = [fonts] * len(lines)
    hs = [size(draw, line, fnt)[1] for line, fnt in zip(lines, fonts)]
    total = sum(hs) + gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2 - 2
    for line, fnt, h in zip(lines, fonts, hs):
        w, _ = size(draw, line, fnt)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + gap


def arrow(draw, start, end, color="#3c4b5f", width=5):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = 17
    spread = 0.55
    p1 = (ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread))
    p2 = (ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def poly(draw, points, color="#3c4b5f", width=5, arrow_end=True):
    draw.line(points, fill=color, width=width)
    if arrow_end:
        arrow(draw, points[-2], points[-1], color, width)


def node(draw, box, title, sub=""):
    draw.rounded_rectangle(box, radius=10, fill="#eaf2fd", outline="#2f6fe4", width=3)
    lines = [title]
    fonts = [F_BOX]
    if sub:
        sub_lines = wrap(draw, sub, F_SUB, box[2] - box[0] - 32)[:2]
        lines += sub_lines
        fonts += [F_SUB] * len(sub_lines)
    center(draw, box, lines, fonts)


def diamond(draw, center_pt, wh, text):
    cx, cy = center_pt
    w, h = wh
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(pts, fill="#3e74c7", outline="#183c73")
    draw.line(pts + [pts[0]], fill="#183c73", width=3)
    center(draw, (cx - w / 2 + 10, cy - h / 2 + 8, cx + w / 2 - 10, cy + h / 2 - 8), wrap(draw, text, F_SUB, w - 25), F_SUB, fill="white")


def callout(draw, box, text, target, side):
    draw.rounded_rectangle(box, radius=8, fill="#fff8f8", outline="#d52121", width=3)
    center(draw, box, wrap(draw, text, F_CALLOUT, box[2] - box[0] - 25)[:2], F_CALLOUT)
    x1, y1, x2, y2 = box
    start = {
        "left": (x1, (y1 + y2) / 2),
        "right": (x2, (y1 + y2) / 2),
        "top": ((x1 + x2) / 2, y1),
        "bottom": ((x1 + x2) / 2, y2),
    }[side]
    arrow(draw, start, target, color="#c00000", width=5)


def draw_fig():
    img = Image.new("RGB", (2200, 1360), "#f7fbff")
    draw = ImageDraw.Draw(img)

    left_x, left_w, left_h = 230, 430, 100
    left_y = [210, 440, 670, 900]
    left_nodes = [
        ("生成权重向量", "角点强化权重"),
        ("建立邻域关系", "B(i)"),
        ("初始化种群", "A*、进度层—横向带扰动、随机初始化"),
        ("初始化理想点", "z* = (z1*, z2*, z3*)"),
    ]
    for y, (title, sub) in zip(left_y, left_nodes):
        node(draw, (left_x, y, left_x + left_w, y + left_h), title, sub)
    for i in range(3):
        arrow(draw, (left_x + left_w / 2, left_y[i] + left_h), (left_x + left_w / 2, left_y[i + 1]))

    mid_x, mid_w = 930, 520
    mid_y = [325, 525, 725, 925]
    stop = (1160, 175)
    diamond(draw, stop, (260, 130), "满足停止\n条件")
    node(draw, (1640, 110, 2145, 250), "输出 Pareto 路径集", "外部非支配档案 A")
    arrow(draw, (stop[0] + 130, stop[1]), (1640, 180))
    draw.text((1205, 255), "否", font=F_BOX, fill="#1f2d3d")
    arrow(draw, (1160, 240), (1160, mid_y[0]))

    mid_nodes = [
        ("选择子问题 i", "基于效用值的主动子问题采样"),
        ("从 B(i) 选父代", "邻域选择降低搜索空间"),
        ("交叉/变异产生候选路径 y", "片段交叉与高斯扰动变异"),
        ("计算 F(y), V(y)", "三目标值、硬/软约束违背度"),
        ("更新理想点/邻域/Pareto 解", "约束优先比较与档案维护"),
    ]
    for idx, (title, sub) in enumerate(mid_nodes):
        y = mid_y[idx] if idx < 4 else 1100
        h = 118 if idx < 4 else 150
        node(draw, (mid_x, y, mid_x + mid_w, y + h), title, sub)
        if idx < 4:
            nxt = mid_y[idx + 1] if idx < 3 else 1100
            arrow(draw, (mid_x + mid_w / 2, y + h), (mid_x + mid_w / 2, nxt))

    poly(draw, [(left_x + left_w / 2, left_y[-1] + left_h), (left_x + left_w / 2, 1200), (800, 1200), (800, 175), (1030, 175)])
    poly(draw, [(mid_x + mid_w / 2, 1250), (mid_x + mid_w / 2, 1310), (800, 1310), (800, 175), (1030, 175)], arrow_end=False)
    arrow(draw, (800, 175), (1030, 175))

    callout(draw, (25, 185, 205, 285), "角点强化权重", (left_x, left_y[0] + 50), "right")
    callout(draw, (25, 665, 205, 765), "混合初始化策略", (left_x, left_y[2] + 55), "right")
    callout(draw, (1400, 40, 1710, 140), "MTOE 提前终止", (stop[0] + 115, stop[1] - 8), "left")
    callout(draw, (1705, 325, 2060, 430), "主动子问题采样机制", (mid_x + mid_w, mid_y[0] + 59), "left")
    callout(draw, (1705, 690, 2060, 795), "定向局部搜索策略", (mid_x + mid_w, mid_y[2] + 80), "left")
    callout(draw, (1705, 1030, 2060, 1135), "外部非支配档案维护", (mid_x + mid_w, 1175), "left")

    FIG32.parent.mkdir(exist_ok=True)
    img.save(FIG32, quality=95)


def find_docx():
    docx = BASE / "论文2.docx"
    if docx.exists():
        return docx
    for path in BASE.glob("*.docx"):
        if path.name.startswith("~") or "before" in path.name:
            continue
        if path.stem.endswith("2"):
            return path
    raise FileNotFoundError("Cannot find thesis docx")


def text(p):
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def replace_docx_image():
    docx = find_docx()
    backup = docx.with_name(f"{docx.stem}_before_adjust_fig3_2_left_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
    shutil.copy2(docx, backup)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with ZipFile(docx, "r") as zin:
            zin.extractall(tmp)
        root = ET.parse(tmp / "word" / "document.xml").getroot()
        rels_root = ET.parse(tmp / "word" / "_rels" / "document.xml.rels").getroot()
        paras = root.findall(".//w:body/w:p", NS)
        caption_idx = None
        for idx, p in enumerate(paras):
            if text(p) == "图 3-2 基于标准 MOEA/D 的改进设计示意图":
                caption_idx = idx
                break
        if caption_idx is None:
            raise RuntimeError("Cannot find 图 3-2 caption")
        rid = None
        for p in reversed(paras[:caption_idx]):
            blip = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
            if blip is not None:
                rid = blip.attrib.get(qn(R_NS, "embed"))
                break
        if not rid:
            raise RuntimeError("Cannot find 图 3-2 image rel")
        target = None
        for rel in rels_root:
            if rel.attrib.get("Id") == rid:
                target = rel.attrib.get("Target")
                break
        if not target:
            raise RuntimeError("Cannot find 图 3-2 image target")
        shutil.copy2(FIG32, tmp / "word" / target)

        out = docx.with_suffix(".tmp.docx")
        if out.exists():
            out.unlink()
        with ZipFile(out, "w", ZIP_DEFLATED) as zout:
            for path in tmp.rglob("*"):
                if path.is_file():
                    zout.write(path, path.relative_to(tmp).as_posix())
        with ZipFile(out) as ztest:
            bad = ztest.testzip()
            if bad:
                raise RuntimeError(f"Bad zip member: {bad}")
        out.replace(docx)
    print(f"updated image: {FIG32}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    draw_fig()
    try:
        replace_docx_image()
    except FileNotFoundError:
        print(f"updated image: {FIG32}")
        print("skip docx update: thesis docx not found")
