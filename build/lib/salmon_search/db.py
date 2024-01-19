import os
import sqlite3

import numpy as np
import sqlite_vss

from . import embeddings
from .schemas import ChunkRecord, Resource, chunk_record_factory

SALMON_DIR = os.getcwd()
DB_PATH = os.path.join(SALMON_DIR, 'salmon.db')
# create ~/salmon directory if it doesn't exist
os.makedirs(SALMON_DIR, exist_ok=True)


def add(a, b):
    return a + b


def create_connection(create: bool = False):
    mode = f"rw{create and 'c' or ''}"
    uri = f"file:{DB_PATH}?mode={mode}"
    conn = sqlite3.connect(uri, uri=True)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    return conn


class DbAlreadyExistsException(Exception):
    pass


def create_db():
    if os.path.exists(DB_PATH):
        raise DbAlreadyExistsException('Database already exists.')

    conn = create_connection(True)
    conn.execute('''
        CREATE TABLE if NOT EXISTS resources (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT
        );
    ''')

    conn.execute('''
        CREATE TABLE if NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk TEXT,
            embedding BLOB,
            resource_id INTEGER,
            FOREIGN KEY (resource_id) REFERENCES resources (id)
        );
    ''')

    conn.execute(f'''
        CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(
        chunk_embedding({embeddings.VECTOR_SIZE});
    );
    ''')

    # Create empty chunk embedding to avoid issue with creating empty vss_table
    zero_array = np.zeros(embeddings.VECTOR_SIZE)
    conn.execute('''
        INSERT INTO vss_chunks (chunk_embedding)
            VALUES (?)
            ''', [zero_array])

    conn.commit()
    conn.close()


def resource_exists_by_url(url: str):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM resources WHERE url = ?
    ''', [url])
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None


def save_resource(resource: Resource):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO resources (url, title)
        VALUES (?, ?)
    ''', (resource.url, resource.title))
    resource_id = cursor.lastrowid
    for chunk, embedding in zip(resource.chunks, resource.embeddings):
        cursor.execute('''
            INSERT INTO chunks (chunk, embedding, resource_id)
            VALUES (?, ?, ?)
        ''', [chunk, embedding, resource_id])

    resource.id = resource_id
    cursor.close()
    conn.commit()
    conn.close()


def update_vss_index():
    conn = create_connection()
    conn.execute('''
            INSERT INTO vss_chunks (rowid, chunk_embedding)
                SELECT c.rowid, c.embedding
                FROM chunks c
            WHERE c.rowid > COALESCE((SELECT MAX(v.rowid) FROM vss_chunks v), 0)
        ''')

    conn.commit()
    conn.close()


def get_most_similar_articles_based_on_n_chunks(n: int, query_embedding: np.ndarray) -> list[ChunkRecord]:
    conn = create_connection()
    cursor = create_cursor_with_row_factory(conn)
    cursor.execute('''
    SELECT MIN(distance) dist, c.rowid, c.chunk, r.rowid, r.title, r.url
    FROM vss_chunks v
    JOIN chunks c ON c.rowid = v.rowid
    JOIN resources r ON r.id = c.resource_id
    WHERE vss_search(
      v.chunk_embedding,
      vss_search_params(
        ?,
        ?
      )
    )
    GROUP BY r.rowid
    ORDER BY dist
    ''', [query_embedding, n])
    results: list[ChunkRecord] = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def get_top_n_chunks(n: int, query_embedding: np.ndarray) -> list[ChunkRecord]:
    conn = create_connection()
    cursor = create_cursor_with_row_factory(conn)
    cursor.execute('''
    SELECT distance dist, c.rowid, c.chunk, r.rowid, r.title, r.url
    FROM vss_chunks v
    JOIN chunks c ON c.rowid = v.rowid 
    JOIN resources r ON r.id = c.resource_id
    WHERE vss_search(
      v.chunk_embedding,
      vss_search_params(
        ?1,
        ?2
      )
    )
    ORDER BY dist
    ''', [query_embedding, n])
    results: list[ChunkRecord] = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def create_cursor_with_row_factory(conn):
    cursor = conn.cursor()
    cursor.row_factory = chunk_record_factory
    return cursor


def get_resource(resource_id: int) -> Resource:
    conn = create_connection()
    cursor = conn.cursor()

    def row_factory(cursor, row):
        return Resource.from_args(*row)

    cursor.row_factory = row_factory

    cursor.execute('''
    SELECT rowid, url, title
    FROM resources
    WHERE rowid = ?
    ''', [resource_id])
    return cursor.fetchone()


def delete_resource(resource_id: int):
    conn = create_connection()
    cursor = conn.cursor()
    resource = get_resource(resource_id)
    cursor.execute(
        '''
        DELETE FROM vss_chunks
        WHERE rowid IN (
            SELECT rowid
            FROM chunks
            WHERE resource_id = ?
        )
        ''',
        [resource_id]
    )
    cursor.execute(
        '''
        DELETE FROM chunks
        WHERE resource_id = ?
        ''',
        [resource_id]
    )
    cursor.execute(
        '''
        DELETE FROM resources
        WHERE rowid = ?
        ''', [resource_id]
    ).fetchone()
    cursor.close()
    conn.commit()
    conn.close()
    return resource


def get_chunk(chunk_id: int) -> str:
    conn = create_connection()
    cursor = conn.cursor()
    chunk = cursor.execute(
        '''
        SELECT chunk
        FROM chunks
        WHERE rowid = ?
        ''',
        [chunk_id]
    ).fetchone()
    if chunk:
        return chunk[0]
    else:
        return f'No chunk with ID {chunk_id}'
