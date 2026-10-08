"""
Small layout layer for the status deck (2026-10-08 refresh).

Slides are described once (rectangles, wrapped text, tables, images) in a
top-left coordinate system, then drawn by two backends:
  * reportlab  -> status_deck.pdf  (same pipeline, palette and built-in
                  Helvetica/Courier fonts as build_pdf_2026-10-01.py)
  * PIL        -> PNG previews, used only to check legibility and overflow
                  (Liberation Sans / Nimbus Mono are metric-compatible with
                  Helvetica / Courier).
Text is wrapped here with reportlab's string widths, so a block that does not
fit its box is detected at build time (OVERFLOWS) instead of being clipped.
Only characters of the WinAnsi set are used (no arrows or math symbols).
"""
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import HexColor

W, H = 960.0, 540.0
OVERFLOWS = []

NAVY, CARD, CARD2 = "#0F172A", "#1E293B", "#14212C"
GREEN, ORANGE, AMBER, BLUE = "#4ADE80", "#F97316", "#FBBF24", "#38BDF8"
INK, INK2, MUTED, FOOTER, RULE = "#F8FAFC", "#E2E8F0", "#94A3B8", "#64748B", "#334155"


def _font(bold, mono):
    if mono:
        return "Courier-Bold" if bold else "Courier"
    return "Helvetica-Bold" if bold else "Helvetica"


def wrap(txt, width, size, bold=False, mono=False):
    fn = _font(bold, mono)
    lines = []
    for para in str(txt).split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            t = (cur + " " + w) if cur else w
            if stringWidth(t, fn, size) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


class Page:
    def __init__(self, name):
        self.name, self.ops = name, []
        self.rect(0, 0, W, H, NAVY)

    # ---- primitives -------------------------------------------------
    def rect(self, x, y, w, h, fill, r=0, stroke=None):
        self.ops.append(("rect", x, y, w, h, fill, r, stroke))

    def line(self, x1, y1, x2, y2, color=RULE, lw=1):
        self.ops.append(("line", x1, y1, x2, y2, color, lw))

    def image(self, x, y, w, h, path):
        from PIL import Image
        iw, ih = Image.open(path).size
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        self.ops.append(("img", x + (w - dw) / 2, y + (h - dh) / 2, dw, dh, path))

    def text(self, x, y, w, txt, size=10, color=INK2, bold=False, mono=False,
             leading=None, maxh=None, align="l"):
        """Wrapped text; y is the TOP of the block; returns its height."""
        lead = leading or size * 1.28
        lines = wrap(txt, w, size, bold, mono)
        h = len(lines) * lead
        if maxh is not None and h > maxh + 0.5:
            OVERFLOWS.append("%s: text block needs %.0f pt, box %.0f pt: %.40s" % (self.name, h, maxh, txt))
        if x + w > W - 20 + 0.5 or y + h > H + 0.5:
            OVERFLOWS.append("%s: text outside page: %.40s" % (self.name, txt))
        for i, ln in enumerate(lines):
            self.ops.append(("text", x, y + size * 0.82 + i * lead, w, ln, size, color, bold, mono, align))
        return h

    def bullets(self, x, y, w, items, size=10, color=INK2, gap=3, maxh=None, mark="-"):
        yy = y
        ind = size * 1.1
        for it in items:
            h = self.text(x + ind, yy, w - ind, it, size, color)
            self.ops.append(("text", x, yy + size * 0.82, ind, mark, size, GREEN, True, False, "l"))
            yy += h + gap
        if maxh is not None and yy - y - gap > maxh + 0.5:
            OVERFLOWS.append("%s: bullets need %.0f pt, box %.0f pt: %.40s" % (self.name, yy - y - gap, maxh, items[0]))
        return yy - y - gap

    def table(self, x, y, colw, rows, size=8.5, header=True, maxh=None, pad=4,
              colors=None, mono_cols=(), bold_cols=(), hdr_fill=CARD, zebra=True):
        """rows: list of lists of str or (str, hexcolor). Returns height."""
        yy = y
        for ri, row in enumerate(rows):
            lines_n, cells = 1, []
            for ci, cell in enumerate(row):
                txt, col = (cell if isinstance(cell, tuple) else (cell, None))
                isb = (header and ri == 0) or ci in bold_cols
                ls = wrap(txt, colw[ci] - 2 * pad, size, isb, ci in mono_cols)
                cells.append((txt, col, ls, isb))
                lines_n = max(lines_n, len(ls))
            lead = size * 1.25
            rh = lines_n * lead + 2 * pad - 1
            if header and ri == 0:
                self.rect(x, yy, sum(colw), rh, hdr_fill)
            elif zebra and ri % 2 == 0:
                self.rect(x, yy, sum(colw), rh, CARD2)
            cx = x
            for ci, (txt, col, ls, isb) in enumerate(cells):
                c = col or (GREEN if (header and ri == 0) else INK2)
                for li, ln in enumerate(ls):
                    self.ops.append(("text", cx + pad, yy + pad + size * 0.82 + li * lead,
                                     colw[ci], ln, size, c, isb, ci in mono_cols, "l"))
                cx += colw[ci]
            yy += rh
        if maxh is not None and yy - y > maxh + 0.5:
            OVERFLOWS.append("%s: table needs %.0f pt, box %.0f pt" % (self.name, yy - y, maxh))
        return yy - y

    # ---- backends ---------------------------------------------------
    def to_pdf(self, c):
        for op in self.ops:
            k = op[0]
            if k == "rect":
                _, x, y, w, h, fill, r, stroke = op
                c.setFillColor(HexColor(fill))
                if stroke:
                    c.setStrokeColor(HexColor(stroke)); c.setLineWidth(1)
                if r:
                    c.roundRect(x, H - y - h, w, h, r, stroke=1 if stroke else 0, fill=1)
                else:
                    c.rect(x, H - y - h, w, h, stroke=1 if stroke else 0, fill=1)
            elif k == "line":
                _, x1, y1, x2, y2, col, lw = op
                c.setStrokeColor(HexColor(col)); c.setLineWidth(lw)
                c.line(x1, H - y1, x2, H - y2)
            elif k == "img":
                _, x, y, w, h, p = op
                c.drawImage(p, x, H - y - h, w, h)
            elif k == "text":
                _, x, yb, w, ln, size, col, bold, mono, al = op
                c.setFillColor(HexColor(col)); c.setFont(_font(bold, mono), size)
                if al == "r":
                    c.drawRightString(x + w, H - yb, ln)
                elif al == "c":
                    c.drawCentredString(x + w / 2, H - yb, ln)
                else:
                    c.drawString(x, H - yb, ln)
        c.showPage()

    def to_png(self, path, scale=2):
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("RGB", (int(W * scale), int(H * scale)), NAVY)
        d = ImageDraw.Draw(im)
        fdir, mdir = "/usr/share/fonts/liberation-sans/LiberationSans-", "/usr/share/fonts/urw-base35/NimbusMonoPS-"
        cache = {}

        def font(size, bold, mono):
            key = (size, bold, mono)
            if key not in cache:
                p = (mdir + ("Bold" if bold else "Regular") + ".otf") if mono else (fdir + ("Bold" if bold else "Regular") + ".ttf")
                cache[key] = ImageFont.truetype(p, size * scale)
            return cache[key]
        for op in self.ops:
            k = op[0]
            if k == "rect":
                _, x, y, w, h, fill, r, stroke = op
                b = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
                if r:
                    d.rounded_rectangle(b, radius=r * scale, fill=fill, outline=stroke)
                else:
                    d.rectangle(b, fill=fill, outline=stroke)
            elif k == "line":
                _, x1, y1, x2, y2, col, lw = op
                d.line([x1 * scale, y1 * scale, x2 * scale, y2 * scale], fill=col, width=max(1, int(lw * scale)))
            elif k == "img":
                _, x, y, w, h, p = op
                pic = Image.open(p).convert("RGB").resize((int(w * scale), int(h * scale)))
                im.paste(pic, (int(x * scale), int(y * scale)))
            elif k == "text":
                _, x, yb, w, ln, size, col, bold, mono, al = op
                f = font(size, bold, mono)
                tw = d.textlength(ln, font=f) / scale
                xx = x + (w - tw if al == "r" else (w - tw) / 2 if al == "c" else 0)
                d.text((xx * scale, yb * scale), ln, font=f, fill=col, anchor="ls")
        im.save(path)


# ---- shared slide furniture ------------------------------------------------
MARGIN = 36


def header(p, eyebrow, title, subtitle=None, tsize=24):
    p.text(MARGIN, 24, W - 2 * MARGIN, eyebrow, 9, GREEN, True, True)
    y = 40
    h = p.text(MARGIN, y, W - 2 * MARGIN, title, tsize, INK, True, leading=tsize * 1.1, maxh=tsize * 2.3)
    y += h + 4
    if subtitle:
        y += p.text(MARGIN, y, W - 2 * MARGIN, subtitle, 10, MUTED, maxh=40) + 4
    return y + 4


def footer(p, n, src):
    p.text(MARGIN, 512, W - 2 * MARGIN - 70, "Sources: " + src, 7.2, FOOTER, maxh=22, leading=8.6)
    p.text(W - MARGIN - 64, 512, 64, "%d | 2026-10-08" % n, 7.5, FOOTER, align="r")


def card(p, x, y, w, h, bar=GREEN, fill=CARD2, r=8):
    p.rect(x, y, w, h, fill, r)
    p.rect(x, y + 3, 4, h - 6, bar)


def banner(p, x, y, w, h, txt, color=ORANGE, size=10):
    p.rect(x, y, w, h, CARD2, 8)
    p.rect(x, y, 5, h, color)
    p.text(x + 16, y + (h - size * 1.28 * len(wrap(txt, w - 30, size, True))) / 2, w - 30, txt, size, color, True, maxh=h)
