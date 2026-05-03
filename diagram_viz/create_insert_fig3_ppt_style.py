from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


BASE = Path(".")
DOCX = BASE / "论文2.docx"
ASSET_DIR = BASE / "paper_assets"
ASSET_DIR.mkdir(exist_ok=True)
FIG31 = ASSET_DIR / "fig_3_1_moead_basic_flow.png"
FIG32 = ASSET_DIR / "fig_3_2_moead_enhanced_design.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W_NS, "r": R_NS, "wp": WP_NS, "a": A_NS, "pic": PIC_NS}
for prefix, uri in [("w", W_NS), ("r", R_NS), ("wp", WP_NS), ("a", A_NS), ("pic", PIC_NS)]:
    ET.register_namespace(prefix, uri)
ET.register_namespace("", CT_NS)


def qn(ns, tag):
    return f"{{{ns}}}{tag}"


def get_font(size, bold=False, italic=False):
    candidates = []
    if italic:
        candidates.append(r"C:\Windows\Fonts\msyhi.ttc")
    candidates += [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FT_TITLE = get_font(58, True)
FT_BOX = get_font(30, True)
FT_SUB = get_font(25)
FT_MATH = get_font(34, False, True)
FT_MATH_SMALL = get_font(22, False, True)
FT_SMALL = get_font(22)


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw, text, font, max_w):
    out = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            cand = line + ch
            if text_size(draw, cand, font)[0] <= max_w or not line:
                line = cand
            else:
                out.append(line)
                line = ch
        if line:
            out.append(line)
    return out


def centered(draw, box, lines, fonts, fill="#1f2d3d", gap=8):
    x1, y1, x2, y2 = box
    if not isinstance(fonts, list):
        fonts = [fonts] * len(lines)
    hs = [text_size(draw, line, font)[1] for line, font in zip(lines, fonts)]
    total = sum(hs) + gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2 - 3
    for line, font, h in zip(lines, fonts, hs):
        w, _ = text_size(draw, line, font)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=font, fill=fill)
        y += h + gap


def rect_node(draw, box, title, sub="", fill="#eaf2fd", outline="#2f6fe4", title_font=FT_BOX):
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=3)
    lines = [title]
    fonts = [title_font]
    if sub:
        lines += wrap(draw, sub, FT_SUB, box[2] - box[0] - 34)[:2]
        fonts += [FT_SUB] * (len(lines) - 1)
    centered(draw, box, lines, fonts)


def arrow(draw, start, end, color="#3c4b5f", width=5):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    ang = math.atan2(ey - sy, ex - sx)
    length = 18
    spread = 0.55
    p1 = (ex - length * math.cos(ang - spread), ey - length * math.sin(ang - spread))
    p2 = (ex - length * math.cos(ang + spread), ey - length * math.sin(ang + spread))
    draw.polygon([end, p1, p2], fill=color)


def polyline(draw, points, color="#3c4b5f", width=5, arrow_end=True):
    draw.line(points, fill=color, width=width)
    if arrow_end and len(points) >= 2:
        arrow(draw, points[-2], points[-1], color, width)


def draw_math_formula(draw, x, y, segments):
    cursor = x
    for text, mode in segments:
        font = FT_MATH_SMALL if mode in {"sup", "sub"} else FT_MATH
        yoff = -15 if mode == "sup" else (18 if mode == "sub" else 0)
        draw.text((cursor, y + yoff), text, font=font, fill="#1f2d3d")
        cursor += text_size(draw, text, font)[0]


def draw_max_with_under(draw, x, y):
    max_text = "max"
    under_text = "1≤i≤3"
    max_w = text_size(draw, max_text, FT_MATH)[0]
    under_w = text_size(draw, under_text, FT_MATH_SMALL)[0]
    draw.text((x, y), max_text, font=FT_MATH, fill="#1f2d3d")
    draw.text((x + (max_w - under_w) / 2, y + 36), under_text, font=FT_MATH_SMALL, fill="#1f2d3d")
    return x + max_w + 18


def diamond(draw, center, size, text):
    cx, cy = center
    w, h = size
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(pts, fill="#3e74c7", outline="#183c73")
    draw.line(pts + [pts[0]], fill="#183c73", width=3)
    centered(draw, (cx - w / 2 + 15, cy - h / 2 + 10, cx + w / 2 - 15, cy + h / 2 - 10), wrap(draw, text, FT_SUB, w - 30), FT_SUB, fill="white")


def make_fig31():
    img = Image.new("RGB", (2200, 1360), "#f7fbff")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 2200, 1360), fill="#f7fbff")

    left_x, left_w, box_h = 130, 380, 92
    left_y = [210, 430, 650, 870]
    left_nodes = [
        ("生成权重向量", "λ1, ..., λN"),
        ("建立邻域关系", "B(i)"),
        ("初始化种群", "f(p1), ..., f(pN)"),
        ("初始化理想点", "z* = (z1*, z2*, z3*)"),
    ]
    for y, (t, s) in zip(left_y, left_nodes):
        rect_node(draw, (left_x, y, left_x + left_w, y + box_h), t, s, fill="#eaf2fd")
    for i in range(3):
        arrow(draw, (left_x + left_w / 2, left_y[i] + box_h), (left_x + left_w / 2, left_y[i + 1]))

    mid_x, mid_w = 880, 460
    mid_y = [315, 515, 715, 915]
    mid_nodes = [
        ("选择子问题 i", "从邻域或全局中选取当前子问题"),
        ("从 B(i) 选父代", "邻域选择降低搜索空间"),
        ("交叉/变异产生新解 y", "得到候选路径 y"),
        ("计算 F(y), V(y)", "目标值与约束违背"),
        ("更新理想点/邻域/Pareto 解", "满足标量化准则则替换"),
    ]
    stop_c = (1110, 170)
    diamond(draw, stop_c, (250, 130), "满足停止\n条件")
    rect_node(draw, (1590, 110, 2090, 250), "输出 Pareto 解与对应路径", "", fill="#edf4ff")
    draw.text((1540, 430), "Tchebycheff 标量化函数：", font=FT_BOX, fill="#1f2d3d")
    draw_math_formula(draw, 1540, 505, [
        ("g", "base"), ("te", "sup"), ("(P | λ", "base"), ("j", "sup"), (", z", "base"), ("*", "sup"), (") =", "base"),
    ])
    formula_x = draw_max_with_under(draw, 1540, 570)
    draw_math_formula(draw, formula_x, 575, [
        ("{ λ", "base"), ("j", "sup"), ("i", "sub"),
        (" | f", "base"), ("i", "sub"), ("(P) - z", "base"), ("*", "sup"), ("i", "sub"), (" | }", "base"),
    ])

    arrow(draw, (stop_c[0] + 125, stop_c[1]), (1590, 180))
    draw.text((1155, 250), "否", font=FT_BOX, fill="#1f2d3d")
    arrow(draw, (1110, 235), (1110, mid_y[0]))
    for idx, (t, s) in enumerate(mid_nodes):
        y = mid_y[idx] if idx < 4 else 1090
        h = 112 if idx < 4 else 150
        rect_node(draw, (mid_x, y, mid_x + mid_w, y + h), t, s, fill="#eaf2fd")
        if idx < 4:
            arrow(draw, (mid_x + mid_w / 2, y + h), (mid_x + mid_w / 2, mid_y[idx + 1] if idx < 3 else 1120))

    polyline(draw, [(left_x + left_w / 2, left_y[-1] + box_h), (left_x + left_w / 2, 1180), (740, 1180), (740, 170), (985, 170)])
    polyline(draw, [(mid_x + mid_w / 2, 1270), (mid_x + mid_w / 2, 1305), (740, 1305), (740, 170), (985, 170)], arrow_end=False)
    arrow(draw, (740, 170), (985, 170))
    img.save(FIG31, quality=95)


def red_callout(draw, box, text, target, side="left"):
    draw.rounded_rectangle(box, radius=8, fill="#fff7f7", outline="#d52121", width=3)
    centered(draw, box, wrap(draw, text, FT_SUB, box[2] - box[0] - 18), FT_SUB, fill="#1f2d3d")
    x1, y1, x2, y2 = box
    if side == "left":
        start = (x1, (y1 + y2) / 2)
    elif side == "right":
        start = (x2, (y1 + y2) / 2)
    elif side == "top":
        start = ((x1 + x2) / 2, y1)
    elif side == "bottom":
        start = ((x1 + x2) / 2, y2)
    else:
        start = ((x1 + x2) / 2, (y1 + y2) / 2)
    arrow(draw, start, target, color="#c00000", width=6)


def make_fig32():
    img = Image.new("RGB", (2200, 1360), "#f7fbff")
    draw = ImageDraw.Draw(img)

    left_x, left_w, box_h = 150, 380, 92
    left_y = [220, 440, 660, 880]
    left_nodes = [
        ("生成权重向量", "λ1, ..., λN"),
        ("建立邻域关系", "B(i)"),
        ("初始化种群", "f(p1), ..., f(pN)"),
        ("初始化理想点", "z* = (z1*, z2*, z3*)"),
    ]
    for y, (t, s) in zip(left_y, left_nodes):
        rect_node(draw, (left_x, y, left_x + left_w, y + box_h), t, s, fill="#eaf2fd")
    for i in range(3):
        arrow(draw, (left_x + left_w / 2, left_y[i] + box_h), (left_x + left_w / 2, left_y[i + 1]))

    mid_x, mid_w = 900, 460
    mid_y = [325, 525, 725, 925]
    stop_c = (1130, 175)
    diamond(draw, stop_c, (250, 130), "满足停止\n条件")
    rect_node(draw, (1600, 110, 2100, 250), "输出 Pareto 解与对应路径", "", fill="#edf4ff")
    arrow(draw, (stop_c[0] + 125, stop_c[1]), (1600, 180))
    draw.text((1175, 255), "否", font=FT_BOX, fill="#1f2d3d")
    arrow(draw, (1130, 240), (1130, mid_y[0]))

    mid_nodes = [
        ("选择子问题 i", "主动子问题采样"),
        ("从 B(i) 选父代", "邻域选择降低搜索空间"),
        ("交叉/变异产生新解 y", "局部搜索增强候选路径"),
        ("计算 F(y), V(y)", "目标值与约束违背"),
        ("更新理想点/邻域/Pareto 解", "外部档案维护增强"),
    ]
    for idx, (t, s) in enumerate(mid_nodes):
        y = mid_y[idx] if idx < 4 else 1090
        h = 112 if idx < 4 else 150
        rect_node(draw, (mid_x, y, mid_x + mid_w, y + h), t, s, fill="#eaf2fd")
        if idx < 4:
            arrow(draw, (mid_x + mid_w / 2, y + h), (mid_x + mid_w / 2, mid_y[idx + 1] if idx < 3 else 1130))

    polyline(draw, [(left_x + left_w / 2, left_y[-1] + box_h), (left_x + left_w / 2, 1190), (760, 1190), (760, 175), (1005, 175)])
    polyline(draw, [(mid_x + mid_w / 2, 1280), (mid_x + mid_w / 2, 1310), (760, 1310), (760, 175), (1005, 175)], arrow_end=False)
    arrow(draw, (760, 175), (1005, 175))

    red_callout(draw, (30, 70, 330, 170), "角点强化权重设计", (left_x, left_y[0] + 46), side="right")
    red_callout(draw, (20, 770, 320, 855), "A* 引导混合初始化", (left_x, left_y[2] + 70), side="right")
    red_callout(draw, (1375, 40, 1675, 140), "MTOE 提前终止", (stop_c[0] + 112, stop_c[1] - 8), side="left")
    red_callout(draw, (1665, 325, 1985, 425), "主动子问题采样", (mid_x + mid_w, mid_y[0] + 56), side="left")
    red_callout(draw, (1665, 665, 1985, 765), "局部搜索增强", (mid_x + mid_w, mid_y[2] + 56), side="left")
    red_callout(draw, (1665, 990, 1985, 1090), "外部档案维护增强", (mid_x + mid_w, mid_y[3] + 56), side="left")
    img.save(FIG32, quality=95)


def p_text(p):
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def paragraph(text, align=None):
    p = ET.Element(qn(W_NS, "p"))
    ppr = ET.SubElement(p, qn(W_NS, "pPr"))
    if align:
        jc = ET.SubElement(ppr, qn(W_NS, "jc"))
        jc.set(qn(W_NS, "val"), align)
    r = ET.SubElement(p, qn(W_NS, "r"))
    t = ET.SubElement(r, qn(W_NS, "t"))
    t.text = text
    return p


def image_paragraph(rel_id, name, cx, cy, docpr_id):
    p = ET.Element(qn(W_NS, "p"))
    ppr = ET.SubElement(p, qn(W_NS, "pPr"))
    jc = ET.SubElement(ppr, qn(W_NS, "jc"))
    jc.set(qn(W_NS, "val"), "center")
    r = ET.SubElement(p, qn(W_NS, "r"))
    drawing = ET.SubElement(r, qn(W_NS, "drawing"))
    inline = ET.SubElement(drawing, qn(WP_NS, "inline"))
    inline.set("distT", "0")
    inline.set("distB", "0")
    inline.set("distL", "0")
    inline.set("distR", "0")
    extent = ET.SubElement(inline, qn(WP_NS, "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect = ET.SubElement(inline, qn(WP_NS, "effectExtent"))
    for key in ("l", "t", "r", "b"):
        effect.set(key, "0")
    docpr = ET.SubElement(inline, qn(WP_NS, "docPr"))
    docpr.set("id", str(docpr_id))
    docpr.set("name", name)
    ET.SubElement(inline, qn(WP_NS, "cNvGraphicFramePr"))
    graphic = ET.SubElement(inline, qn(A_NS, "graphic"))
    data = ET.SubElement(graphic, qn(A_NS, "graphicData"))
    data.set("uri", PIC_NS)
    pic = ET.SubElement(data, qn(PIC_NS, "pic"))
    nv = ET.SubElement(pic, qn(PIC_NS, "nvPicPr"))
    cnv = ET.SubElement(nv, qn(PIC_NS, "cNvPr"))
    cnv.set("id", "0")
    cnv.set("name", name)
    ET.SubElement(nv, qn(PIC_NS, "cNvPicPr"))
    fill = ET.SubElement(pic, qn(PIC_NS, "blipFill"))
    blip = ET.SubElement(fill, qn(A_NS, "blip"))
    blip.set(qn(R_NS, "embed"), rel_id)
    stretch = ET.SubElement(fill, qn(A_NS, "stretch"))
    ET.SubElement(stretch, qn(A_NS, "fillRect"))
    sppr = ET.SubElement(pic, qn(PIC_NS, "spPr"))
    xfrm = ET.SubElement(sppr, qn(A_NS, "xfrm"))
    off = ET.SubElement(xfrm, qn(A_NS, "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(xfrm, qn(A_NS, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geom = ET.SubElement(sppr, qn(A_NS, "prstGeom"))
    geom.set("prst", "rect")
    ET.SubElement(geom, qn(A_NS, "avLst"))
    return p


def next_rel_id(root):
    vals = []
    for rel in root:
        rid = rel.attrib.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            vals.append(int(rid[3:]))
    return f"rId{max(vals, default=0) + 1}"


def max_docpr_id(root):
    vals = []
    for item in root.findall(".//wp:docPr", NS):
        val = item.attrib.get("id")
        if val and val.isdigit():
            vals.append(int(val))
    return max(vals, default=0)


def next_image_index(media_dir):
    nums = []
    for path in media_dir.glob("image*.*"):
        m = re.match(r"image(\d+)", path.stem)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def ensure_png_content_type(root):
    for child in root:
        if child.tag == qn(CT_NS, "Default") and child.attrib.get("Extension") == "png":
            return
    item = ET.Element(qn(CT_NS, "Default"))
    item.set("Extension", "png")
    item.set("ContentType", "image/png")
    root.insert(0, item)


def insert():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DOCX.with_name(f"论文2_before_insert_fig3_pptstyle_{stamp}.docx")
    shutil.copy2(DOCX, backup)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with ZipFile(DOCX) as zin:
            zin.extractall(tmp)
        doc_path = tmp / "word" / "document.xml"
        rel_path = tmp / "word" / "_rels" / "document.xml.rels"
        ct_path = tmp / "[Content_Types].xml"
        doc_tree = ET.parse(doc_path)
        doc_root = doc_tree.getroot()
        rel_tree = ET.parse(rel_path)
        rel_root = rel_tree.getroot()
        ct_tree = ET.parse(ct_path)
        ct_root = ct_tree.getroot()
        body = doc_root.find("w:body", NS)
        paras = body.findall("w:p", NS)

        chapter3 = None
        section32 = None
        for i, p in enumerate(paras):
            txt = p_text(p)
            if i > 200 and txt.startswith("第三章 基于 MOEA/D"):
                chapter3 = p
            if i > 200 and txt == "3.2 面向任务特性的增强设计":
                section32 = p
        if chapter3 is None or section32 is None:
            raise RuntimeError("未找到第三章正文或 3.2 标题")

        media_dir = tmp / "word" / "media"
        media_dir.mkdir(exist_ok=True)
        docpr = max_docpr_id(doc_root)
        figs = []
        for fig_path, title in [
            (FIG31, "图 3-1 MOEA/D 基础流程图"),
            (FIG32, "图 3-2 基于标准 MOEA/D 的改进设计示意图"),
        ]:
            idx = next_image_index(media_dir)
            target = media_dir / f"image{idx}.png"
            shutil.copy2(fig_path, target)
            rid = next_rel_id(rel_root)
            rel = ET.SubElement(rel_root, qn(REL_NS, "Relationship"))
            rel.set("Id", rid)
            rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            rel.set("Target", f"media/{target.name}")
            im = Image.open(fig_path)
            cx = int(15.9 / 2.54 * 914400)
            cy = int(cx * im.height / im.width)
            docpr += 1
            figs.append((rid, target.name, title, cx, cy, docpr))

        block31 = [
            paragraph("为说明 MOEA/D 在多目标路径规划中的基本执行逻辑，图 3-1 给出了标准 MOEA/D 的基础流程。算法首先生成权重向量并建立邻域关系，在初始化种群和理想点后进入迭代搜索；每轮迭代根据停止条件判断是否结束，若未满足终止条件，则选择子问题、从邻域中选取父代、通过交叉变异产生新解，并计算候选路径的目标函数值和约束违背度。"),
            image_paragraph(figs[0][0], figs[0][1], figs[0][3], figs[0][4], figs[0][5]),
            paragraph(figs[0][2], align="center"),
            paragraph("由图 3-1 可知，MOEA/D 的核心在于利用标量化函数将多目标优化问题分解为多个具有不同偏好的子问题，并通过邻域协同更新逐步逼近 Pareto 前沿。该流程为本文后续引入路径规划任务相关的初始化、采样、局部搜索、档案维护和终止控制机制提供了基础框架。"),
        ]
        block32 = [
            paragraph("在标准 MOEA/D 流程基础上，本文进一步结合复杂环境路径规划任务特点进行改进，图 3-2 给出了主要增强机制与基础流程之间的对应关系。"),
            image_paragraph(figs[1][0], figs[1][1], figs[1][3], figs[1][4], figs[1][5]),
            paragraph(figs[1][2], align="center"),
            paragraph("如图 3-2 所示，角点强化权重设计用于增强极端目标方向的覆盖能力，A* 引导混合初始化用于提高初始可行解比例，主动子问题采样用于将搜索资源分配到更具改进潜力的子问题，局部搜索增强用于改善候选路径的局部质量，外部档案维护增强用于保存非支配代表解，MTOE 提前终止则用于减少后期无效迭代。这些改进共同服务于复杂环境下 Pareto 路径集的可行性、收敛性、多样性和计算效率。"),
        ]

        children = list(body)
        ch_idx = children.index(chapter3)
        intro_idx = None
        for idx in range(ch_idx + 1, len(children)):
            if children[idx].tag == qn(W_NS, "p"):
                intro_idx = idx
                break
        if intro_idx is None:
            raise RuntimeError("未找到第三章引言段落")
        for offset, elem in enumerate(block31):
            body.insert(intro_idx + 1 + offset, elem)

        children = list(body)
        section32 = None
        for child in children:
            if child.tag == qn(W_NS, "p") and p_text(child) == "3.2 面向任务特性的增强设计":
                section32 = child
        if section32 is None:
            raise RuntimeError("未重新找到 3.2 标题")
        s32_idx = list(body).index(section32)
        for offset, elem in enumerate(block32):
            body.insert(s32_idx + 1 + offset, elem)

        ensure_png_content_type(ct_root)
        doc_tree.write(doc_path, encoding="UTF-8", xml_declaration=True)
        rel_tree.write(rel_path, encoding="UTF-8", xml_declaration=True)
        ct_tree.write(ct_path, encoding="UTF-8", xml_declaration=True)

        out = DOCX.with_suffix(".tmp.docx")
        if out.exists():
            out.unlink()
        with ZipFile(out, "w", ZIP_DEFLATED) as zout:
            for file in tmp.rglob("*"):
                if file.is_file():
                    zout.write(file, file.relative_to(tmp).as_posix())
        with ZipFile(out) as ztest:
            bad = ztest.testzip()
            if bad:
                raise RuntimeError(f"docx zip 检查失败: {bad}")
        out.replace(DOCX)
    print(f"created: {FIG31}")
    print(f"created: {FIG32}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    make_fig31()
    make_fig32()
    if DOCX.exists():
        insert()
    else:
        print(f"created: {FIG31}")
        print(f"created: {FIG32}")
        print(f"skip docx insert: {DOCX} not found")
