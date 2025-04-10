from bs4 import BeautifulSoup
from mkdocs.config import Config
from mkdocs.structure.files import Files
from pathlib import Path

import os

def prettify_site(site_dir):
	for html_file in site_dir.rglob("*.html"):
		prettyify_html(html_file)

def prettyify_html(html_file):
	with open(html_file, "r", encoding = "utf-8") as f:
		soup = BeautifulSoup(f, "html.parser")

	with open(html_file, "w", encoding = "utf-8") as f:
		f.write(soup.prettify())

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
