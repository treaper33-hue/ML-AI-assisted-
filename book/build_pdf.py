"""Renders the book's Markdown chapters into a typeset PDF.

Pipeline: Markdown -> HTML (python-markdown + Pygments) -> PDF (headless
Chrome). No LaTeX, no pandoc, no wkhtmltopdf: the only non-stdlib pieces
are `markdown` and `pygments`, both already present in the environment the
testbed runs in.

Usage:
    python build_pdf.py              # chapters -> build/book.html -> book.pdf
    python build_pdf.py --html-only  # stop at HTML, for fast iteration
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
from typing import List
from typing import Optional

import markdown

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / 'build'
OUT_PDF = HERE.parent / 'Machine_Learning_The_Working_Text.pdf'

CHROME_CANDIDATES = (
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
)

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
@page :first { margin: 0; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Cambria", serif;
  font-size: 10.5pt; line-height: 1.52; color: #15181d; margin: 0;
  hyphens: auto;
}

/* ---- cover ------------------------------------------------------- */
.cover {
  height: 292mm; box-sizing: border-box; padding: 55mm 22mm 20mm 22mm;
  background: #11161f; color: #f3f5f8; page-break-after: always;
  overflow: hidden;
}
.cover h1 {
  font-size: 34pt; line-height: 1.1; margin: 0 0 6mm 0; color: #fff;
  border: none; padding: 0; font-weight: 700; letter-spacing: -0.5px;
  /* The global rule below breaks a page before every h1; the cover's
     own title must not trigger it, or page one renders empty. */
  page-break-before: avoid;
}
.cover .sub { font-size: 14pt; color: #9fb3d0; margin-bottom: 28mm; }
.cover .meta { font-size: 9.5pt; color: #8c9bb0; line-height: 1.9; }
.cover .rule { width: 42mm; height: 3px; background: #4b89d6; margin: 0 0 8mm 0; }

/* ---- headings ---------------------------------------------------- */
h1 {
  font-size: 20pt; margin: 0 0 7mm 0; padding-bottom: 3mm;
  border-bottom: 2.5px solid #11161f; page-break-before: always;
  page-break-after: avoid; letter-spacing: -0.3px;
}
h1.nobreak { page-break-before: avoid; }
h2 {
  font-size: 14pt; margin: 9mm 0 3mm 0; color: #11161f;
  page-break-after: avoid; border-bottom: 1px solid #d4dae3;
  padding-bottom: 1.5mm;
}
h3 {
  font-size: 11.5pt; margin: 6mm 0 2mm 0; color: #26303e;
  page-break-after: avoid;
}
h4 {
  font-size: 10.5pt; margin: 5mm 0 1.5mm 0; color: #3d4858;
  font-style: italic; page-break-after: avoid;
}
p { margin: 0 0 3mm 0; text-align: justify; }

/* ---- code -------------------------------------------------------- */
code, pre, .codehilite {
  font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
}
p code, li code, td code {
  font-size: 8.8pt; background: #eef1f6; padding: 0.4mm 1mm;
  border-radius: 2px; color: #11304d;
}
.codehilite, pre {
  background: #f7f9fc; border: 1px solid #dce3ec; border-left: 3px solid #4b89d6;
  border-radius: 3px; padding: 3mm 4mm; font-size: 8.5pt; line-height: 1.42;
  overflow-x: hidden; page-break-inside: avoid; margin: 3mm 0 4mm 0;
}
.codehilite pre { background: none; border: none; padding: 0; margin: 0; }
pre code { font-size: 8.5pt; white-space: pre-wrap; word-wrap: break-word; }

/* ---- display equations ------------------------------------------- */
.eq {
  display: block; text-align: center; margin: 4mm 0; padding: 2.5mm 3mm;
  background: #fbfbfd; border-top: 1px solid #e3e8ef;
  border-bottom: 1px solid #e3e8ef; font-size: 11pt;
  font-family: "Cambria Math", "Cambria", "Georgia", serif;
  page-break-inside: avoid;
}

/* ---- tables ------------------------------------------------------ */
table {
  border-collapse: collapse; width: 100%; margin: 3mm 0 5mm 0;
  font-size: 8.8pt; page-break-inside: avoid;
}
th {
  background: #11161f; color: #fff; text-align: left; padding: 2mm 2.5mm;
  font-weight: 600;
}
td { border-bottom: 1px solid #dce3ec; padding: 1.8mm 2.5mm; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fc; }

/* ---- callouts ---------------------------------------------------- */
blockquote {
  margin: 4mm 0; padding: 3mm 4mm; background: #f2f6fd;
  border-left: 3px solid #4b89d6; page-break-inside: avoid;
}
blockquote p { margin: 0 0 2mm 0; }
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }

hr { border: none; border-top: 1px solid #d4dae3; margin: 6mm 0; }

/* ---- table of contents ------------------------------------------- */
.toc-page { page-break-after: always; }
.toc ul { list-style: none; padding-left: 4mm; margin: 0; }
.toc > ul { padding-left: 0; }
.toc li { margin-bottom: 1mm; font-size: 9.5pt; }
.toc > ul > li { margin-top: 2.5mm; font-weight: 600; font-size: 10.5pt; }
.toc a { color: #11161f; text-decoration: none; }

a { color: #1a4b86; text-decoration: none; }

/* ---- source anchors ---------------------------------------------- */
.anchor-note {
  font-size: 8.5pt; color: #5a6878; font-style: italic;
  border-top: 1px dotted #c8d1dd; padding-top: 1.5mm; margin-top: 2mm;
}
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
<style>{pygments}</style>
</head>
<body>
{body}
</body>
</html>
"""


def chapter_files() -> List[pathlib.Path]:
    """Returns the chapter sources in reading order.

    Returns:
        Sorted list of Markdown files; the numeric prefix sets the order.
    """
    return sorted(HERE.glob('[0-9][0-9]_*.md'))


def pygments_css() -> str:
    """Returns the syntax-highlighting stylesheet.

    Returns:
        A CSS string, or an empty string if Pygments is unavailable.
    """
    try:
        from pygments.formatters import HtmlFormatter
    except ImportError:
        return ''
    return HtmlFormatter(style='friendly').get_style_defs('.codehilite')


def render_html(sources: List[pathlib.Path]) -> str:
    """Converts the chapter sources into one HTML document.

    Args:
        sources: Markdown files in reading order.

    Returns:
        The complete HTML document as a string.

    Raises:
        FileNotFoundError: If no chapter sources were found.
    """
    if not sources:
        raise FileNotFoundError(f'no chapters matching [0-9][0-9]_*.md in {HERE}')
    text = '\n\n'.join(path.read_text(encoding='utf-8') for path in sources)
    converter = markdown.Markdown(
        extensions=[
            'extra', 'tables', 'fenced_code', 'codehilite', 'toc',
            'sane_lists', 'attr_list', 'admonition',
        ],
        extension_configs={
            'codehilite': {'css_class': 'codehilite', 'guess_lang': False},
            'toc': {'title': '', 'toc_depth': '1-2'},
        },
    )
    body = converter.convert(text)
    toc = getattr(converter, 'toc', '')
    body = body.replace(
        '<!--TOC-->',
        f'<div class="toc-page"><h1 class="nobreak">Contents</h1>'
        f'<div class="toc">{toc}</div></div>',
    )
    return HTML_TEMPLATE.format(
        title='Machine Learning: The Working Text',
        css=CSS,
        pygments=pygments_css(),
        body=body,
    )


def find_chrome() -> Optional[str]:
    """Locates a Chromium-family browser able to print to PDF.

    Returns:
        A path to the executable, or None when none is installed.
    """
    for candidate in CHROME_CANDIDATES:
        if pathlib.Path(candidate).exists():
            return candidate
    for name in ('chrome', 'msedge', 'chromium'):
        found = shutil.which(name)
        if found:
            return found
    return None


def print_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path) -> bool:
    """Prints an HTML file to PDF with headless Chrome.

    Args:
        html_path: Source HTML.
        pdf_path: Destination PDF.

    Returns:
        True when the PDF was produced.
    """
    browser = find_chrome()
    if not browser:
        print('no Chromium-family browser found; HTML written instead')
        return False
    profile = BUILD / 'chrome-profile'
    command = [
        browser,
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        '--no-pdf-header-footer',
        '--run-all-compositor-stages-before-draw',
        '--virtual-time-budget=20000',
        f'--user-data-dir={profile}',
        f'--print-to-pdf={pdf_path}',
        html_path.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if not pdf_path.exists():
        print('chrome did not produce a PDF')
        print((result.stderr or '')[-800:])
        return False
    return True


def main() -> int:
    """Entry point.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html-only', action='store_true',
                        help='stop after writing the HTML')
    args = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    sources = chapter_files()
    html = render_html(sources)
    html_path = BUILD / 'book.html'
    html_path.write_text(html, encoding='utf-8')
    words = sum(len(p.read_text(encoding='utf-8').split()) for p in sources)
    print(f'chapters : {len(sources)}')
    print(f'words    : {words:,}')
    print(f'html     : {html_path}')

    if args.html_only:
        return 0
    if print_pdf(html_path, OUT_PDF):
        size_kb = OUT_PDF.stat().st_size / 1024
        print(f'pdf      : {OUT_PDF} ({size_kb:,.0f} KB)')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
