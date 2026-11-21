"""Book import pipeline for the Onyx Studio reader.

Turns one source document (TXT / Markdown / EPUB) into a paged book directory:

    <out-dir>/book.json    metadata, reading progress, typography settings
    <out-dir>/pages.json   chapter index + one record per page (text + prompt)
    <out-dir>/source.txt   normalized source text

Standard library only — runs under the engine venv's python (3.8+ compatible).
The host passes the raw document bytes through a payload file so any size or
encoding survives the command line:

    python book_import.py --format txt --payload-file /tmp/x.payload \
        --book-id b-xyz --title "Journey to the West" --author "Wu Cheng’en" --style ink \
        --chars-per-page 800 --out-dir artifacts/studio/books/b-xyz

The last stdout line is a JSON summary document; progress lines go to stderr.
"""
import argparse
import html as html_mod
import json
import os
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

CHAPTER_RE = re.compile(
    r"^(第[0-9〇零一二两三四五六七八九十百千万]+[回章节部幕卷集]"
    r"|卷[0-9〇零一二两三四五六七八九十百千万]+"
    r"|楔子|序章|序言|自序|前言|导论|后记|尾声|终章|番外)"
    r"([\s：:．.、].*)?$"
)
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;…\n])")
SCENE_KEYWORDS = (
    "山", "水", "天", "月", "风", "云", "雨", "雪", "花", "林", "河", "海", "火", "灯",
    "桥", "船", "马", "剑", "刀", "旗", "城", "殿", "宫", "庙", "洞", "石", "树", "星",
    "夜", "酒", "门", "路", "楼", "窗", "庭", "院", "江湖", "舟", "兵", "将", "帅",
    "僧", "道", "佛", "仙", "怪", "妖", "鬼", "神", "府",
)
AD_LINE_RE = re.compile(
    r"(笔趣阁|www\.|http://|https://|\.com|\.net|\.cc|首发|防盗|请记住|"
    r"最新章节|本书来自|章节错误|精校版|无错版|更多精彩|未完待续|本章未完|点击下一页|继续阅读)"
)
CLASSIC_HINT_RE = re.compile(r"(西游记|红楼梦|水浒传|三国演义|三国志演义|石头记|金瓶梅|聊斋|封神)")

STYLE_PRESETS = {
    "ink": {
        "zh": "古典水墨",
        "classical": "ancient Chinese classical setting, Ming-dynasty clothing and architecture",
        "suffix": "Traditional Chinese ink-wash painting (shui-mo), expressive brush strokes, misty negative space, muted ink tones with one accent color, rice paper texture, no text, no watermark",
    },
    "cinematic": {
        "zh": "写实电影感",
        "classical": "ancient Chinese classical setting, period-accurate costume and architecture",
        "suffix": "photorealistic cinematic film still, dramatic volumetric lighting, shallow depth of field, 35mm, rich color grading, highly detailed, no text, no watermark",
    },
    "storybook": {
        "zh": "插画绘本",
        "classical": "",
        "suffix": "warm storybook illustration, soft painterly brushwork, gentle palette, cozy lighting, hand-drawn texture, no text, no watermark",
    },
    "minimal": {
        "zh": "极简图形",
        "classical": "",
        "suffix": "minimalist flat illustration, clean geometry, limited palette, generous negative space, elegant composition, no text, no watermark",
    },
}

PUNCT_END = tuple("。！？!?…”’」』）)；;：:—..")


# ----------------------------------------------------------------- decoding

def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# -------------------------------------------------------------- txt / md

def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.strip().replace("\u3000", " ").strip()
        if not line:
            lines.append("")
            continue
        if AD_LINE_RE.search(line) and len(line) < 60 and not CHAPTER_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def merge_wrapped_lines(text: str) -> list:
    """Join hard-wrapped Chinese lines back into paragraphs."""
    paragraphs = []
    buffer = ""
    for line in text.split("\n"):
        if not line:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        if CHAPTER_RE.match(line) and len(line) < 60:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(line)
            continue
        if buffer:
            buffer += line
        else:
            buffer = line
        if line.endswith(PUNCT_END) or len(line) > 200:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def split_markdown(text: str) -> list:
    chapters = []
    title = ""
    buffer = []
    for line in text.split("\n"):
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            if buffer:
                chapters.append((title, list(buffer)))
                buffer = []
            title = match.group(2).strip() or "Untitled"
        else:
            buffer.append(line)
    if buffer:
        chapters.append((title, list(buffer)))
    return [(t or "Untitled", ps) for t, ps in chapters if "".join(ps).strip()]


# ------------------------------------------------------------------- epub

class BodyText(HTMLParser):
    BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "section", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs = []
        self.title = ""
        self._buffer = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.split(":")[-1].lower()
        if tag in ("h1", "h2", "h3") and not self.title:
            self._in_title = True
        elif tag == "br":
            self.flush()

    def handle_endtag(self, tag):
        tag = tag.split(":")[-1].lower()
        if self._in_title and tag in ("h1", "h2", "h3"):
            self._in_title = False
        if tag in self.BLOCKS:
            self.flush()

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
            return
        self._buffer += text

    def flush(self):
        if self._buffer.strip():
            self.paragraphs.append(self._buffer.strip())
        self._buffer = ""


def epub_documents(raw: bytes):
    """Yield (chapter_title_hint, paragraphs) per spine document."""
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        opf_path = rootfile.get("full-path")
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(archive.read(opf_path))
        manifest = {}
        for item in opf.findall(".//{*}manifest/{*}item"):
            manifest[item.get("id")] = (item.get("href"), item.get("media-type", ""))
        book_title = opf.findtext(".//{*}title") or ""
        spine_ids = [ref.get("idref") for ref in opf.findall(".//{*}spine/{*}itemref")]
        for index, idref in enumerate(spine_ids):
            entry = manifest.get(idref)
            if entry is None:
                continue
            href, media_type = entry
            if media_type and "html" not in media_type and "xml" not in media_type:
                continue
            doc_path = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else posixpath.normpath(href)
            try:
                doc_raw = archive.read(doc_path)
            except KeyError:
                continue
            parser = BodyText()
            try:
                parser.feed(decode_text(doc_raw))
                parser.close()
            except Exception:
                continue
            parser.flush()
            paragraphs = [p for p in parser.paragraphs if len(p) > 0]
            if not paragraphs:
                continue
            hint = parser.title or os.path.splitext(posixpath.basename(doc_path))[0]
            yield book_title, index, hint, paragraphs


def split_epub(raw: bytes, book_title: str) -> list:
    chapters = []
    for default_title, index, hint, paragraphs in epub_documents(raw):
        clean = html_mod.unescape(hint).strip() or f"{default_title} {index + 1}"
        if re.fullmatch(r"[0-9]+|chapter[_\-\s]?[0-9]+", clean, re.IGNORECASE):
            clean = f"{default_title} {index + 1}" if default_title else clean
        chapters.append((clean[:60], paragraphs))
    return chapters


# ------------------------------------------------------------- pagination

def split_long(paragraph: str, limit: int) -> list:
    if len(paragraph) <= limit:
        return [paragraph]
    pieces = [p for p in SENTENCE_RE.split(paragraph) if p]
    chunks = []
    buffer = ""
    for piece in pieces:
        if len(piece) > limit:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for offset in range(0, len(piece), limit):
                chunks.append(piece[offset:offset + limit])
            continue
        if buffer and len(buffer) + len(piece) > limit:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer += piece
    if buffer:
        chunks.append(buffer)
    return chunks


def paginate(paragraphs: list, limit: int) -> list:
    pages = []
    current = []
    length = 0
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in split_long(paragraph, limit):
            if current and length + len(chunk) > limit:
                pages.append(current)
                current = []
                length = 0
            current.append(chunk)
            length += len(chunk) + 1
    if current:
        pages.append(current)
    return pages


# ---------------------------------------------------------------- prompts

def scene_sentences(page_text: str, limit: int = 140) -> str:
    sentences = [s.strip() for s in SENTENCE_RE.split(page_text) if len(s.strip()) >= 10]
    scored = []
    for sentence in sentences:
        hits = sum(1 for keyword in SCENE_KEYWORDS if keyword in sentence)
        length_score = 1.0 if 12 <= len(sentence) <= 60 else 0.4
        scored.append((hits * length_score, sentence))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    picked = []
    total = 0
    for score, sentence in scored:
        if score <= 0 or len(picked) >= 2:
            break
        if total + len(sentence) > limit:
            continue
        picked.append(sentence)
        total += len(sentence)
    if not picked and sentences:
        picked = [sentences[0][:80]]
    if not picked:
        picked = [page_text[:80]]
    return "，".join(sentence.rstrip("。，；！？、」』）") for sentence in picked)


def build_prompt(page_text: str, chapter_title: str, style_id: str, classical: bool) -> str:
    """Scene-only prompt. Chapter titles are deliberately excluded: image
    models love rendering any text they are given as calligraphy."""
    style = STYLE_PRESETS.get(style_id, STYLE_PRESETS["ink"])
    scene = scene_sentences(page_text)
    parts = [scene]
    if classical and style["classical"]:
        parts.append(style["classical"])
    parts.append(style["suffix"])
    return ", ".join(parts)


# ------------------------------------------------------------------- main

def run(args) -> dict:
    raw = open(args.payload_file, "rb").read()
    if args.format == "epub":
        raw_text = ""
        try:
            chapters = split_epub(raw, args.title)
        except Exception as error:
            raise ValueError(f"Could not parse EPUB ({error}). Check whether the file is encrypted or damaged.") from error
        if not chapters:
            raise ValueError("No readable content was found in the EPUB spine.")
        classical = bool(args.title and CLASSIC_HINT_RE.search(args.title)) or any(
            "第" in title and "回" in title for title, _ in chapters[:8])
    else:
        raw_text = decode_text(raw)
        normalized = normalize_text(raw_text)
        if args.format == "md":
            chapters = split_markdown(normalized)
            classical = bool(args.title and CLASSIC_HINT_RE.search(args.title))
        else:
            paragraphs = merge_wrapped_lines(normalized)
            chapters = []
            current_title = "Opening"
            current = []
            for paragraph in paragraphs:
                if CHAPTER_RE.match(paragraph) and len(paragraph) < 60:
                    if current:
                        chapters.append((current_title, current))
                    current_title = paragraph
                    current = []
                else:
                    current.append(paragraph)
            if current:
                chapters.append((current_title, current))
            chapters = [(t, ps) for t, ps in chapters if sum(len(p) for p in ps) > 0]
            if not chapters:
                raise ValueError("No chapters were found. Check the encoding or try Markdown or EPUB.")
            heading_hits = sum(1 for title, _ in chapters if CHAPTER_RE.match(title))
            classical = bool(args.title and CLASSIC_HINT_RE.search(args.title)) or heading_hits >= 2

    pages = []
    chapter_index = []
    for index, (title, paragraphs) in enumerate(chapters):
        first_page = len(pages)
        for page_paragraphs in paginate(paragraphs, args.chars_per_page):
            text = "\n".join(page_paragraphs)
            pages.append({
                "chapter": index,
                "chapterTitle": title,
                "text": text,
                "prompt": build_prompt(text, title, args.style, classical),
                "image": None,
            })
        chapter_index.append({"index": index, "title": title, "firstPage": first_page})

    char_count = sum(len(page["text"]) for page in pages)
    if char_count == 0:
        raise ValueError("The parsed document is empty.")

    book_id = args.book_id
    os.makedirs(args.out_dir, exist_ok=True)
    if raw_text:
        with open(os.path.join(args.out_dir, "source.txt"), "w", encoding="utf-8") as handle:
            handle.write(raw_text)
    else:
        with open(os.path.join(args.out_dir, "source.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"[binary source: {args.format}, {len(raw)} bytes]")
    with open(os.path.join(args.out_dir, "pages.json"), "w", encoding="utf-8") as handle:
        json.dump({"chapters": chapter_index, "pages": pages}, handle, ensure_ascii=False)
    meta = {
        "id": book_id,
        "title": args.title or "Untitled",
        "author": args.author or "",
        "format": args.format,
        "style": args.style,
        "classical": classical,
        "charsPerPage": args.chars_per_page,
        "pageCount": len(pages),
        "chapterCount": len(chapters),
        "charCount": char_count,
        "createdAtMs": int(os.path.getmtime(args.payload_file) * 1000),
        "updatedAtMs": int(os.path.getmtime(args.payload_file) * 1000),
        "progress": {"lastPage": 0},
        "settings": {
            "theme": "paper",
            "fontSize": 19,
            "lineHeight": 1.9,
            "serif": "song",
            "autoIllustrate": True,
            "aheadCount": 2,
        },
    }
    with open(os.path.join(args.out_dir, "book.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)
    return {
        "ok": True,
        "bookId": book_id,
        "title": meta["title"],
        "pageCount": len(pages),
        "chapterCount": len(chapters),
        "charCount": char_count,
        "classical": classical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("txt", "md", "epub"), required=True)
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--author", default="")
    parser.add_argument("--style", choices=tuple(STYLE_PRESETS), default="ink")
    parser.add_argument("--chars-per-page", type=int, default=800)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:  # user-facing diagnostics travel as one JSON line
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
