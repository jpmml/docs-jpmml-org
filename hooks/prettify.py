from bs4 import BeautifulSoup
from bs4.builder import HTMLParserTreeBuilder
from mkdocs.config import Config
from mkdocs.structure.files import Files
from pathlib import Path

import os
import re

class CodePreservingTreeBuilder(HTMLParserTreeBuilder):
    DEFAULT_PRESERVE_WHITESPACE_TAGS = HTMLParserTreeBuilder.DEFAULT_PRESERVE_WHITESPACE_TAGS | {"code"}

def prettify_site(site_dir):
	for html_file in site_dir.rglob("*.html"):
		prettyify_html(html_file)

def prettyify_html(html_file):
	with open(html_file, "r", encoding = "utf-8") as f:
		soup = BeautifulSoup(f, "html.parser", builder = CodePreservingTreeBuilder())

	pretty_html = soup.prettify(formatter = "minimal")

	# Remove before punctuation
	pretty_html = re.sub(r"</code>\s*\n\s*([.,;:\)\]\}!?])", r"</code>\1", pretty_html)

	# Replace with a single space before text
	pretty_html = re.sub(r"</code>\s*\n\s*(\S)", r"</code> \1", pretty_html)

	with open(html_file, "w", encoding = "utf-8") as f:
		f.write(pretty_html)

def on_files(files: Files, config: Config):
	site_dir = Path(config["site_dir"])
	prettify_site(site_dir)
	return files

def on_post_build(config: Config):
	site_dir = Path(config["site_dir"])
	prettify_site(site_dir)

if __name__ == "__main__":
	site_dir = Path("site")
	prettify_site(site_dir)
