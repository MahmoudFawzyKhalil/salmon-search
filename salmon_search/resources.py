import re
from urllib.parse import urlparse

import bs4
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

from . import embeddings
from .schemas import Resource

YOUTUBE_DESCRIPTION_PATTERN = re.compile('(?<=shortDescription":").*(?=","isCrawlable)')
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)


def download_youtube_video_title_and_text_chunks(url):
    # source: https://stackoverflow.com/a/72355455/15763551
    html = requests.get(url).content
    soup = bs4.BeautifulSoup(html, 'html.parser')
    description = YOUTUBE_DESCRIPTION_PATTERN.findall(str(soup))[0].replace('\\n', '\n')
    return soup.find('title').text, [description]


def is_youtube_video(url):
    url_parts = urlparse(url)
    return url_parts.netloc.find("youtu.be") != -1 \
        or url_parts.netloc.find("youtube") != -1


def create_resource(url: str) -> Resource:
    resource = Resource(url)

    if is_youtube_video(url):
        resource.title, resource.chunks = download_youtube_video_title_and_text_chunks(url)
    else:
        resource.title, resource.chunks = download_article_title_and_text_chunks(url)

    resource.embeddings = embeddings.encode(resource.chunks, show_progress_bar=True)
    return resource


def download_article_title_and_text_chunks(url: str) -> tuple[str, list[str]]:
    # Download HTML article
    html = requests.get(url).content
    # Parse HTML
    soup = bs4.BeautifulSoup(html, 'html.parser')
    chunks = SPLITTER.split_text(soup.get_text().replace("\n\n\n", ""))
    return soup.find('title').text, chunks

    # Consider: index all code examples separately by grabbing pre tags?
