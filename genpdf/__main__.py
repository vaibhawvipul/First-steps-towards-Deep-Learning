"""
To run:
Install requirements using `pip install -r requirements.txt`, then:
 1. Open root directory in terminal
 2. Run command `python -m genpdf`
"""

import re
from pathlib import Path

from markdown2 import Markdown
from weasyprint import HTML

THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT_DIR = THIS_DIR.parent
OUTPUT_FILENAME = PROJECT_ROOT_DIR / 'First-Steps-Towards-Deep-Learning.pdf'
CSS_FILENAME = THIS_DIR / 'assets/css/pdf.css'

def replace_images(html):
    html = re.sub('https://github.com/.*/blob/master', PROJECT_ROOT_DIR.as_uri(), html)
    return html


def get_cover_image():
    images = PROJECT_ROOT_DIR.glob('images/*.png')
    cover_img = None
    for im in images:
        if 'cover' in im.name.lower():
            cover_img = im.as_uri()
            return f'<img src="{cover_img}">'


def main():
    chapter_folders = [d for d in PROJECT_ROOT_DIR.iterdir() if d.is_dir() and d.name.lower().startswith('chapter')]
    chapter_folders.sort()
    csstyle = '<link rel="stylesheet" type="text/css" href="genpdf/assets/css/pdf.css">'
    pretag = '\n<p style="page-break-before: always" >\n'
    endtag = '\n</p>\n'
    markdowner = Markdown(extras=['fenced-code-blocks'])
    cover = get_cover_image()
    all_html = [cover] if cover else []
    all_html.append(csstyle)
    for fold in chapter_folders:
        md = next(fold.glob('*.md'))
        with open(md, 'r', encoding='utf-8') as f:
            content = f.read()
            html = markdowner.convert(content)
            html = replace_images(html)
            all_html.append(pretag + html + endtag)

    all_html = '<script></script>'.join(all_html)
    HTML(string=all_html).write_pdf(OUTPUT_FILENAME, stylesheets=[CSS_FILENAME])
    print("Successfully cooked the book!")

if __name__ == '__main__':
    main()
