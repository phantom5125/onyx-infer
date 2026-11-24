#!/usr/bin/env python3
"""Build the Onyx Infer blog from Markdown sources.

Reads   site/blog/posts/*.md  (front matter + a small Markdown subset)
Writes  site/blog/<slug>.html and site/blog/index.html

Zero dependencies: Python standard library only.

Usage:
    uv run python tools/build_blog.py            # build
    uv run python tools/build_blog.py --watch    # rebuild on change
"""
from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "site" / "blog" / "posts"
OUT_DIR = ROOT / "site" / "blog"
SITE_URL = "https://phantom5125.github.io/onyx-infer"
GITHUB = "https://github.com/phantom5125/onyx-infer"

# --------------------------------------------------------------------------
# chrome
# --------------------------------------------------------------------------

MARK_SVG = """<svg viewBox="0 0 64 64" width="26" height="26" aria-hidden="true"><defs><linearGradient id="m{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff7a6e"/><stop offset="0.5" stop-color="#e5484d"/><stop offset="1" stop-color="#8f1220"/></linearGradient></defs><polygon points="32,4 56,18 56,46 32,60 8,46 8,18" fill="#131722" stroke="url(#m{uid})" stroke-width="3"/><polygon points="32,18 44,25 44,39 32,46 20,39 20,25" fill="url(#m{uid})" opacity="0.85"/></svg>"""


def chrome(rel: str, title: str, description: str, og_image: str | None, body: str, uid: str, *, lang: str = "en") -> str:
    og = ""
    if og_image:
        og = (
            f'<meta property="og:image" content="{SITE_URL}/{og_image}">\n'
            f'<meta name="twitter:card" content="summary_large_image">\n'
        )
    return f"""<!doctype html>
<html lang="{html.escape(lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
{og}<link rel="icon" type="image/svg+xml" href="{rel}assets/onyx-mark.svg">
<link rel="stylesheet" href="{rel}assets/style.css">
</head>
<body>
<nav class="nav"><div class="nav-inner">
  <a class="brand" href="{rel}index.html">{MARK_SVG.format(uid=uid)}<span class="word">ONYX <b>INFER</b></span></a>
  <div class="nav-links">
    <a href="{rel}index.html#engine">Engine</a>
    <a href="{rel}index.html#benchmarks">Benchmarks</a>
    <a href="{rel}blog/index.html">Blog</a>
    <a class="gh" href="{GITHUB}" target="_blank" rel="noopener">GitHub &#8599;</a>
  </div>
</div></nav>
{body}
<footer class="site"><div class="wrap">
  <div class="foot-grid">
    <div class="col">
      <h4>Project</h4>
      <a href="{rel}index.html">Home</a>
      <a href="{rel}blog/index.html">Blog</a>
      <a href="{GITHUB}" target="_blank" rel="noopener">GitHub repository</a>
      <a href="{GITHUB}/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">Contributing</a>
    </div>
    <div class="col">
      <h4>Design docs</h4>
      <a href="{GITHUB}/blob/main/docs/ARCHITECTURE.md" target="_blank" rel="noopener">Architecture</a>
      <a href="{GITHUB}/blob/main/docs/PROTOCOLS.md" target="_blank" rel="noopener">Protocols (onyx.*.v1)</a>
      <a href="{GITHUB}/blob/main/docs/LOOP.md" target="_blank" rel="noopener">Optimization loop</a>
      <a href="{GITHUB}/blob/main/docs/V1_REVIEW.md" target="_blank" rel="noopener">v1 review</a>
    </div>
    <div class="col">
      <h4>Evidence</h4>
      <a href="{GITHUB}/blob/main/benchmarks/results/native-v1-stability.json" target="_blank" rel="noopener">native-v1-stability.json</a>
      <a href="{GITHUB}/blob/main/benchmarks/results/native-round4-fused-attention-views.json" target="_blank" rel="noopener">round 4 measurements</a>
      <a href="{GITHUB}/blob/main/benchmarks/results/m5-max-36gb-q4-operator-v1.json" target="_blank" rel="noopener">Q4 operator comparison</a>
    </div>
  </div>
  <p class="foot-note">Onyx Infer source code is available under the <a href="https://github.com/phantom5125/onyx-infer/blob/main/LICENSE">MIT License</a>. &copy; 2026 Onyx Infer contributors.</p>
</div></footer>
</body>
</html>
"""


# --------------------------------------------------------------------------
# markdown subset
# --------------------------------------------------------------------------

CODE_RE = re.compile(r"`([^`]+)`")
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
EM_RE = re.compile(r"\*([^*]+)\*")


def inline(text: str) -> str:
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = CODE_RE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = IMG_RE.sub(lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}">', text)
    text = LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = EM_RE.sub(r"<em>\1</em>", text)
    for i, code in enumerate(codes):
        text = text.replace(
            f"\x00{i}\x00", f"<code>{html.escape(code, quote=False)}</code>"
        )
    return text


def render_markdown(src: str) -> str:
    lines = src.splitlines()
    out: list[str] = []
    para: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            flush()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(body))}</code></pre>")
            i += 1
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        if line.startswith("#### "):
            flush(); out.append(f"<h4>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            flush(); out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            flush(); out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            flush(); out.append(f"<h2>{inline(line[2:])}</h2>")
        elif re.fullmatch(r"-{3,}", line.strip()):
            flush(); out.append('<hr class="endrule">')
        elif line.startswith("> "):
            flush()
            quote: list[str] = []
            while i < len(lines) and lines[i].startswith("> "):
                quote.append(lines[i][2:])
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue
        elif line.startswith("| ") or line.startswith("|-"):
            flush()
            rows: list[list[str]] = []
            header: list[str] | None = None
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    i += 1
                    continue
                if header is None:
                    header = cells
                else:
                    rows.append(cells)
                i += 1
            if header:
                th = "".join(f"<th>{inline(c)}</th>" for c in header)
                body_rows = "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
                    for row in rows
                )
                out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body_rows}</tbody></table>")
            continue
        elif re.match(r"^- ", line):
            flush()
            items: list[str] = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\d+\. ", line):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(f"<li>{inline(re.sub(r'^\d+\. ', '', lines[i]))}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        else:
            para.append(line.strip())

        i += 1

    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------
# front matter
# --------------------------------------------------------------------------

def parse_post(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = src
    if src.startswith("---"):
        end = src.find("\n---", 3)
        if end != -1:
            for line in src[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = src[end + 4:].lstrip("\n")
    required = ("title", "date", "summary")
    missing = [k for k in required if k not in meta]
    if missing:
        sys.exit(f"{path.name}: missing front-matter keys: {', '.join(missing)}")
    meta["slug"] = path.stem
    meta["body_html"] = render_markdown(body)
    return meta


# --------------------------------------------------------------------------
# page templates
# --------------------------------------------------------------------------

def article_page(post: dict) -> str:
    rel = "../"
    hero = ""
    if post.get("image"):
        caption = post.get("caption", "")
        cap = f'<figcaption class="small">{html.escape(caption)}</figcaption>' if caption else ""
        hero = (
            f'<figure class="hero-img"><img src="{rel}{post["image"]}" '
            f'alt="{html.escape(post.get("caption", post["title"]))}">{cap}</figure>'
        )
    kicker = post.get("tags", "Engineering note").replace(",", " · ")
    body = f"""
<main class="article article-wrap">
  <header>
    <div class="kicker">{html.escape(kicker)}</div>
    <h1>{inline(post["title"])}</h1>
    <div class="meta">{post["date"]} &middot; Onyx Infer engineering notes</div>
    {hero}
  </header>
  <div class="prose">
{post["body_html"]}
  </div>
  <p><a class="backlink" href="index.html">&#8592; All posts</a></p>
</main>
"""
    og_image = f"{post['image']}" if post.get("image") else None
    return chrome(rel, f"{post['title']} — Onyx Infer", post["summary"], og_image, body, uid="a",
                  lang=post.get("lang", "en"))


def index_page(posts: list[dict]) -> str:
    rel = "../"
    cards = []
    for p in posts:
        img = f'<img src="{rel}{p["image"]}" alt="">' if p.get("image") else ""
        lang = f' lang="{html.escape(p["lang"])}"' if p.get("lang") else ""
        cards.append(
            f'<a class="post-card" href="{p["slug"]}.html"{lang}>{img}'
            f"<div><div class=\"date\">{p['date']}</div>"
            f"<h3>{inline(p['title'])}</h3>"
            f"<p>{inline(p['summary'])}</p></div></a>"
        )
    body = f"""
<main class="wrap" style="padding-bottom:40px">
  <section style="padding-bottom:0">
    <div class="section-head">
      <div class="eyebrow">Engineering notes</div>
      <h2>Build logs from a native inference engine</h2>
      <p class="lede">Short engineering write-ups on what we changed, what we measured, and how
      each result affected the model output.</p>
    </div>
    <div class="blog-list">
      {''.join(cards)}
    </div>
  </section>
</main>
"""
    return chrome(rel, "Blog — Onyx Infer", "Engineering notes from the Onyx Infer native inference engine.", None, body, uid="i")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    posts = sorted(
        (parse_post(p) for p in sorted(POSTS_DIR.glob("*.md"))),
        key=lambda m: m["date"],
        reverse=True,
    )
    if not posts:
        sys.exit(f"no posts found in {POSTS_DIR}")
    for post in posts:
        page = OUT_DIR / f"{post['slug']}.html"
        page.write_text(article_page(post), encoding="utf-8")
        print(f"  wrote {page.relative_to(ROOT)}")
    index = OUT_DIR / "index.html"
    index.write_text(index_page(posts), encoding="utf-8")
    print(f"  wrote {index.relative_to(ROOT)}  ({len(posts)} posts)")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        print("watching for changes, Ctrl-C to stop…")
        last = 0.0
        while True:
            newest = max((p.stat().st_mtime for p in POSTS_DIR.glob("*.md")), default=0.0)
            if newest > last:
                last = newest or last
                try:
                    build()
                except SystemExit as exc:
                    print(exc)
                last = time.time()
            time.sleep(1.0)
    build()
