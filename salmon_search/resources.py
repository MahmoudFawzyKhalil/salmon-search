import re
from urllib.parse import urlparse

import bs4
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from requests import Response

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
        or url_parts.netloc.find("youtube") != -1 \
        and url_parts.path.find("playlist") == -1


def create_resource(url: str) -> Resource:
    resource = Resource(url)

    if is_youtube_video(url):
        resource.title, resource.chunks = download_youtube_video_title_and_text_chunks(url)
    else:
        resource.title, resource.chunks = download_article_title_and_text_chunks(url)

    if len(resource.chunks) == 0:
        raise Exception(f"No text chunks found for {url}")

    if len(resource.chunks) > 100:
        raise Exception(f"Too many text chunks (> 100) found for {url}")

    resource.embeddings = embeddings.encode(resource.chunks, show_progress_bar=True)
    return resource


def download_article_title_and_text_chunks(url: str) -> tuple[str, list[str]]:
    # Download HTML article
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    }
    print(f"Downloading... {url}")
    res: Response = requests.get(url, headers=headers, timeout=(10, 10), allow_redirects=Truez)
    print(res)
    if res.status_code != 200:
        raise Exception(f"Failed to download article from {url}, status code {res.status_code}")

    html = res.content
    # Parse HTML
    soup = bs4.BeautifulSoup(html, 'html.parser')
    chunks = SPLITTER.split_text(soup.get_text().replace("\n\n\n", ""))
    return extract_title(url, soup), chunks


def extract_title(url: str, soup: bs4.BeautifulSoup) -> str:
    title = soup.find('title')
    if title is not None:
        return title.text
    # Fallback to URL
    return urlparse(url).path
