import re
import math
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from rasterize import rasterize_sketch

# =========================
# 1) Bezier sampling
# =========================

def sample_quadratic(p0, p1, p2, n=25):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts

def sample_cubic(p0, p1, p2, p3, n=25):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = ((1 - t) ** 3) * p0[0] + 3 * ((1 - t) ** 2) * t * p1[0] + 3 * (1 - t) * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = ((1 - t) ** 3) * p0[1] + 3 * ((1 - t) ** 2) * t * p1[1] + 3 * (1 - t) * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        pts.append((x, y))
    return pts

# =========================
# 2) Tokenizer
# =========================

COMMAND_RE = re.compile(r"[MmLlHhVvCcQqSsTtZz]")
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def tokenize_path_d(d: str):
    tokens = []
    i = 0
    while i < len(d):
        ch = d[i]
        if ch.isspace() or ch == ',':
            i += 1
            continue
        if COMMAND_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        m = FLOAT_RE.match(d, i)
        if not m:
            i += 1
            continue
        tokens.append(m.group(0))
        i = m.end()
    return tokens

def is_command(tok) -> bool:
    return isinstance(tok, str) and len(tok) == 1 and COMMAND_RE.match(tok) is not None

def is_number(tok) -> bool:
    return isinstance(tok, str) and not is_command(tok)

# =========================
# 3) Parse one path -> points
# =========================

def parse_path_points(d: str, points_per_curve: int = 24):
    """
    Parse SVG path d and return list of (x,y) points.
    Supports: M/m L/l H/h V/v C/c S/s Q/q T/t Z/z
    """
    tokens = tokenize_path_d(d)
    if not tokens:
        return []

    pts = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)

    last_cubic_ctrl2 = None  # for S/s
    last_quad_ctrl = None    # for T/t

    i = 0
    cmd = None

    def next_float():
        nonlocal i
        if i >= len(tokens) or is_command(tokens[i]):
            raise ValueError(f"Expected number, got {tokens[i] if i < len(tokens) else None}")
        val = float(tokens[i])
        i += 1
        return val

    def reflect(p, around):
        return (2 * around[0] - p[0], 2 * around[1] - p[1])

    while i < len(tokens):
        # read command if present
        if is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            break

        # reset reflection anchors when leaving curve families
        if cmd not in ('C', 'c', 'S', 's'):
            last_cubic_ctrl2 = None
        if cmd not in ('Q', 'q', 'T', 't'):
            last_quad_ctrl = None

        # ---- Move ----
        if cmd in ('M', 'm'):
            x = next_float(); y = next_float()
            if cmd == 'm':
                x += cur[0]; y += cur[1]
            cur = (x, y)
            start = cur
            pts.append(cur)

            # IMPORTANT FIX:
            # implicit L/l only if next token is a number, not a command like 'c'
            if i < len(tokens) and is_number(tokens[i]):
                cmd = 'L' if cmd == 'M' else 'l'
            else:
                cmd = None

        # ---- Line ----
        elif cmd in ('L', 'l'):
            x = next_float(); y = next_float()
            if cmd == 'l':
                x += cur[0]; y += cur[1]
            cur = (x, y)
            pts.append(cur)

        # ---- H/V ----
        elif cmd in ('H', 'h'):
            x = next_float()
            if cmd == 'h':
                x += cur[0]
            cur = (x, cur[1])
            pts.append(cur)

        elif cmd in ('V', 'v'):
            y = next_float()
            if cmd == 'v':
                y += cur[1]
            cur = (cur[0], y)
            pts.append(cur)

        # ---- Cubic ----
        elif cmd in ('C', 'c'):
            x1 = next_float(); y1 = next_float()
            x2 = next_float(); y2 = next_float()
            x  = next_float(); y  = next_float()
            if cmd == 'c':
                x1 += cur[0]; y1 += cur[1]
                x2 += cur[0]; y2 += cur[1]
                x  += cur[0]; y  += cur[1]
            p1 = (x1, y1); p2 = (x2, y2); p3 = (x, y)
            seg = sample_cubic(cur, p1, p2, p3, n=points_per_curve)
            pts.extend(seg[1:])
            cur = p3
            last_cubic_ctrl2 = p2
            last_quad_ctrl = None

        # ---- Smooth Cubic ----
        elif cmd in ('S', 's'):
            if last_cubic_ctrl2 is None:
                p1 = cur
            else:
                p1 = reflect(last_cubic_ctrl2, cur)

            x2 = next_float(); y2 = next_float()
            x  = next_float(); y  = next_float()
            if cmd == 's':
                x2 += cur[0]; y2 += cur[1]
                x  += cur[0]; y  += cur[1]
            p2 = (x2, y2); p3 = (x, y)
            seg = sample_cubic(cur, p1, p2, p3, n=points_per_curve)
            pts.extend(seg[1:])
            cur = p3
            last_cubic_ctrl2 = p2
            last_quad_ctrl = None

        # ---- Quadratic ----
        elif cmd in ('Q', 'q'):
            x1 = next_float(); y1 = next_float()
            x  = next_float(); y  = next_float()
            if cmd == 'q':
                x1 += cur[0]; y1 += cur[1]
                x  += cur[0]; y  += cur[1]
            p1 = (x1, y1); p2 = (x, y)
            seg = sample_quadratic(cur, p1, p2, n=points_per_curve)
            pts.extend(seg[1:])
            cur = p2
            last_quad_ctrl = p1
            last_cubic_ctrl2 = None

        # ---- Smooth Quadratic ----
        elif cmd in ('T', 't'):
            if last_quad_ctrl is None:
                p1 = cur
            else:
                p1 = reflect(last_quad_ctrl, cur)

            x = next_float(); y = next_float()
            if cmd == 't':
                x += cur[0]; y += cur[1]
            p2 = (x, y)
            seg = sample_quadratic(cur, p1, p2, n=points_per_curve)
            pts.extend(seg[1:])
            cur = p2
            last_quad_ctrl = p1
            last_cubic_ctrl2 = None

        # ---- Close ----
        elif cmd in ('Z', 'z'):
            cur = start
            pts.append(cur)
            cmd = None

        else:
            # Not supporting A/a here. Sketchy typically doesn't need it.
            break

    # remove consecutive duplicates
    cleaned = []
    for p in pts:
        if not cleaned or (abs(p[0] - cleaned[-1][0]) > 1e-6 or abs(p[1] - cleaned[-1][1]) > 1e-6):
            cleaned.append(p)
    return cleaned

# =========================
# 4) SVG -> vector sequence
# =========================

_ENTITY_OK = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[A-Za-z_:][\w:.-]*);")

def sanitize_svg_text(text: str) -> str:
    # 1) Xoá comment an toàn (kể cả comment bị lỗi / không đóng)
    #    - xoá các block <!-- ... --> (non-greedy)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    #    - nếu còn comment mở mà không có --> thì xoá từ <!-- tới hết
    text = re.sub(r"<!--.*\Z", "", text, flags=re.DOTALL)

    # 2) Sửa & trần thành &amp; (trừ entity hợp lệ)
    def fix_amp(m):
        s = m.group(0)  # chuỗi bắt đầu từ '&'
        return s if _ENTITY_OK.match(s) else "&amp;" + s[1:]

    # bắt các chuỗi bắt đầu bằng '&' tới trước khoảng trắng/</>"'
    text = re.sub(r"&[^;\s<>\"]*;?", fix_amp, text)

    # 3) Nếu file hay bị thiếu </svg> thì thêm vào cuối (optional nhưng hữu ích)
    if "</svg>" not in text:
        text = text.rstrip() + "\n</svg>\n"

    return text

def svg_to_vector_sequence(
    svg_path: str,
    out_side: int = 224,
    points_per_curve: int = 64,
    padding: float = 8.0,
    clamp: bool = True,
):
    """
    Returns np.ndarray (T,3): [x, y, pen_up]
    - Coordinates are normalized into [0, out_side)
    - pen_up = 1 at the last point of each SVG <path> (stroke)
    """
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        # đọc nội dung file
        with open(svg_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # nếu thiếu </svg> thì thêm vào cuối
        if "</svg>" not in content:
            content = content.strip() + "\n</svg>"

        content = sanitize_svg_text(content)
        # parse lại từ string
        tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    strokes = []
    for elem in root.iter():
        if strip_ns(elem.tag) == "path":
            d = elem.attrib.get("d", "")
            if not d:
                continue
            pts = parse_path_points(d, points_per_curve=points_per_curve)
            if len(pts) >= 2:
                strokes.append(pts)

    if not strokes:
        raise ValueError("No valid <path d='...'> found in SVG.")

    # Flatten to compute bbox
    flat = [p for s in strokes for p in s]
    xs = np.array([p[0] for p in flat], dtype=np.float16)
    ys = np.array([p[1] for p in flat], dtype=np.float16)

    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    # keep aspect ratio and fit to canvas with padding
    scale = min((out_side - 2 * padding) / w, (out_side - 2 * padding) / h)
    new_w = w * scale
    new_h = h * scale

    tx = (out_side - new_w) / 2 - minx * scale
    ty = (out_side - new_h) / 2 - miny * scale

    seq = []
    for stroke in strokes:
        for (x, y) in stroke:
            X = x * scale + tx
            Y = y * scale + ty
            seq.append([X, Y, 0.0])
        seq[-1][2] = 1.0  # pen_up at end stroke

    vector_image = np.array(seq, dtype=np.float16)

    if clamp:
        vector_image[:, 0] = np.clip(vector_image[:, 0], 0, out_side - 1)
        vector_image[:, 1] = np.clip(vector_image[:, 1], 0, out_side - 1)

    return vector_image

vector_image = svg_to_vector_sequence(
    "D:/Research/VLM_project/dataset/Sketchy_FG/sketches/airplane/n02691156_8352-6.svg",
)

sketch_img = rasterize_sketch(vector_image)
sketch_img = 255 - sketch_img
sketch_img = Image.fromarray(sketch_img).convert("RGB")

# Resize về 224x224
sketch_img = sketch_img.resize((224, 224), Image.BILINEAR)

# Lưu ảnh
sketch_img.save("sketch_224.png")