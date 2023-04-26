import collections

from numpy.core.multiarray import ndarray


class Resource:
    def __init__(self, url: str):
        self.id: int | None = None
        self.url = url
        self.chunks: list[str] = []
        self.embeddings: ndarray | None = None
        self.title: str = ''

    @classmethod
    def from_args(cls, *args) -> 'Resource':
        r = Resource(args[1])
        r.id = args[0]
        r.title = args[2]
        return r


ChunkRecord = collections.namedtuple('ChunkRecord', 'distance chunk_id chunk resource_id resource_title resource_url')


def chunk_record_factory(cursor, row):
    return ChunkRecord(*row)
