"""Fixture tests for book_import.py — standalone, stdlib only.

    python3 tests/run_book_import_test.py
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "book_import.py")
PY = sys.executable


def run_import(**kwargs):
    args = [PY, SCRIPT]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, f"importer failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
    return json.loads(lines[-1]), result


def fixture_txt() -> bytes:
    text = """书荒吧？就来笔趣阁 www.biquge.com 看最新章节！
楔子
灵根育孕源流出　心性修持大道生
盖闻天地之数，有十二万九千六百岁为一元。盘古破鸿蒙，开辟从兹清浊辨。
再过五千四百岁，交亥会之初，则当黑暗，而两间昏昏然矣。
第一回　灵根育孕源流出
海外有一国土，名曰傲来国。国近大海，海中有一座名山，唤为花果山。
那座山，正当顶上，有一块仙石。其石有三丈六尺五寸高，按周天三百六十五度；
盖自开辟以来，每受天真地秀，日精月华，感之既久，遂有灵通之意。内育仙胞，一日迸裂，产一石卵，似圆球样大。
石猴在内，目运两道金光，射冲斗府。惊动高天上圣大慈仁者玉皇大天尊玄穹高上帝。
第二回　悟彻菩提真妙理
话表美猴王得遇仙长，需要求道访师。他漂洋过海，登界游方，有十数个年头，
才到了西牛贺洲地界。忽见一座高山，山上有座洞府，洞门紧闭，静悄悄的，正是灵台方寸山斜月三星洞。
樵夫指路，说洞中有一位老神仙，名叫须菩提祖师。美猴王便去敲门求道。
（本章未完，点击下一页继续阅读）"""
    return text.replace("\n", "\n").encode("gb18030")


def fixture_md() -> bytes:
    return """# 计算机网络基础

网络是把独立运行的计算机连接起来的系统。

分层模型把复杂的通信问题拆成可管理的小问题。每一层只关心自己的协议与接口。

## 传输层

传输层为两台主机中的应用进程提供端到端的逻辑通信。TCP 提供可靠字节流，UDP 提供尽力而为的数据报。

TCP 通过序号、确认与重传机制保证可靠传输，通过滑动窗口进行流量控制与拥塞控制。
""".encode("utf-8")


def fixture_epub() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""")
        archive.writestr("OEBPS/content.opf", """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>论文选读</dc:title></metadata>
  <manifest>
    <item id="c1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chap2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>""")
        archive.writestr("OEBPS/chap1.xhtml", """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>引言</title></head><body>
<h1>引言</h1><p>深度学习模型的结构化剪枝能够在保持精度的同时压缩模型。</p>
<p>剪枝方法主要分为结构化与非结构化两类。结构化剪枝直接移除通道或层，硬件友好。</p></body></html>""")
        archive.writestr("OEBPS/chap2.xhtml", """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>方法</title></head><body>
<h1>方法</h1><p>本文提出一种基于重要性评分的通道选择策略。</p><p>评分由梯度与权重的乘积近似。</p></body></html>""")
    return buffer.getvalue()


def check(name, value, expect):
    ok = value == expect or (expect is True and value)
    assert ok, f"{name}: got {value!r}"
    print(f"  ok {name}: {value!r}" if not isinstance(value, str) or len(value) < 60 else f"  ok {name}: {len(value)} chars")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        payload = os.path.join(tmp, "book.payload")
        out_dir = os.path.join(tmp, "book-out")

        # TXT (GBK, hard-wrapped lines, ad lines, 章回 headings)
        with open(payload, "wb") as handle:
            handle.write(fixture_txt())
        summary, _ = run_import(format="txt", payload_file=payload, book_id="b-test",
                                title="西游记", author="吴承恩", style="ink", chars_per_page=120,
                                out_dir=out_dir)
        check("txt chapters", summary["chapterCount"], 3)
        check("txt pages>=4", summary["pageCount"] >= 4, True)
        check("txt classical", summary["classical"], True)
        meta = json.load(open(os.path.join(out_dir, "book.json"), encoding="utf-8"))
        pages_doc = json.load(open(os.path.join(out_dir, "pages.json"), encoding="utf-8"))
        check("meta title", meta["title"], "西游记")
        check("meta pageCount", meta["pageCount"], summary["pageCount"])
        page_texts = [page["text"] for page in pages_doc["pages"]]
        joined = "\n".join(page_texts)
        check("ad line dropped", "笔趣阁" not in joined, True)
        check("ad line dropped 2", "本章未完" not in joined, True)
        check("wrapped merged", "灵根育孕源流出 心性修持大道生盖闻天地之数" in joined, True)
        check("chapter titles", pages_doc["chapters"][1]["title"].startswith("第一回"), True)
        first_page = pages_doc["pages"][0]
        check("page prompt has style suffix", "ink-wash" in first_page["prompt"], True)
        check("page limit respected", all(len(page["text"]) <= 120 * 3 for page in pages_doc["pages"]), True)

        # MD
        with open(payload, "wb") as handle:
            handle.write(fixture_md())
        summary, _ = run_import(format="md", payload_file=payload, book_id="b-md", title="计算机网络",
                                style="minimal", chars_per_page=300, out_dir=os.path.join(tmp, "md-out"))
        check("md chapters", summary["chapterCount"], 2)
        md_pages = json.load(open(os.path.join(tmp, "md-out", "pages.json"), encoding="utf-8"))
        check("md headings as titles", md_pages["chapters"][1]["title"], "传输层")

        # EPUB
        with open(payload, "wb") as handle:
            handle.write(fixture_epub())
        summary, _ = run_import(format="epub", payload_file=payload, book_id="b-epub", title="论文选读",
                                style="cinematic", chars_per_page=400, out_dir=os.path.join(tmp, "epub-out"))
        check("epub chapters", summary["chapterCount"], 2)
        epub_doc = json.load(open(os.path.join(tmp, "epub-out", "pages.json"), encoding="utf-8"))
        check("epub chapter titles", [c["title"] for c in epub_doc["chapters"]], ["引言", "方法"])

        # broken epub must fail with a readable error
        with open(payload, "wb") as handle:
            handle.write(b"not a zip")
        result = subprocess.run([PY, SCRIPT, "--format", "epub", "--payload-file", payload,
                                 "--book-id", "b-bad", "--out-dir", os.path.join(tmp, "bad-out")],
                                capture_output=True, text=True)
        check("broken epub rc", result.returncode == 1, True)
        check("broken epub message", "EPUB" in result.stdout, True)

    print("BOOK-IMPORT-TEST-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
