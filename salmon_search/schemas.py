import collections

from numpy.core.multiarray import ndarray


class Resource:
    def __init__(self, url: str):
        self.id: int | None = None
        self.url = url
        self.chunks: list[str] = []
        self.embeddings: ndarray = []
        self.title: str = ''


ChunkRecord = collections.namedtuple('ChunkRecord', 'distance chunk_id chunk resource_id resource_title resource_url')


def chunk_record_factory(cursor, row):
    return ChunkRecord(*row)
