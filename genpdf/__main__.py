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

THIS_DIR = Path(__file__).parent
PROJECT_ROOT_DIR = THIS_DIR.parent
OUTPUT_FILENAME = PROJECT_ROOT_DIR / 'First-Steps-Towards-Deep-Learning.pdf'
CSS_FILENAME = THIS_DIR / 'pdf.css'


def replace_images(html):
    html = re.sub('https://github.com/.*/blob/master', PROJECT_ROOT_DIR.as_uri(), html)
    return html


def main():
    chapter_folders = [d for d in PROJECT_ROOT_DIR.iterdir() if d.is_dir() and d.name.lower().startswith('chapter')]
    markdowner = Markdown(extras=['fenced-code-blocks'])
    all_html = []
    for fold in chapter_folders:
        md = next(fold.glob('*.md'))
        with open(md, 'r', encoding='utf-8') as f:
            content = f.read()
            html = markdowner.convert(content)
            html = replace_images(html)
            all_html.append(html)

    all_html = ''.join(all_html)
    HTML(string=all_html).write_pdf(OUTPUT_FILENAME, stylesheets=[CSS_FILENAME])


if __name__ == '__main__':
    main()
