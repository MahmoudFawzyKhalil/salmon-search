import os
import sqlite3

import numpy as np
import sqlite_vss

from . import embeddings
from . import schemas


def get_db_dir():
    here = os.path.dirname(__file__)
    db_file = os.path.join(here, '../db/salmon.db')
    db_file = os.path.normpath(db_file)
    return db_file


DB_FILE_PATH = get_db_dir()


def create_connection(create: bool = False):
    mode = f"rw{create and 'c' or ''}"
    uri = f"file:{DB_FILE_PATH}?mode={mode}"
    conn = sqlite3.connect(uri, uri=True)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    return conn


class DbAlreadyExistsException(Exception):
    pass


def create_db():
    if os.path.exists(DB_FILE_PATH):
        raise DbAlreadyExistsException('Database already exists.')

    conn = create_connection(True)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT
        );
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
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
    conn.execute(f'''
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


def save_resource(resource: schemas.Resource):
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


def get_most_similar_articles_based_on_n_chunks(n: int, query_embedding: np.ndarray) -> list[schemas.ChunkRecord]:
    conn = create_connection()
    cursor = create_cursor_with_row_factory(conn)
    cursor.execute('''
    SELECT min(distance) dist, c.rowid, c.chunk, r.rowid, r.title, r.url
    from vss_chunks v
    join chunks c on c.rowid = v.rowid
    join resources r on r.id = c.resource_id
    where vss_search(
      v.chunk_embedding,
      vss_search_params(
        ?,
        ?
      )
    )
    group by r.rowid
    order by dist
    ''', [query_embedding, n])
    results: list[schemas.ChunkRecord] = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def get_top_n_chunks(n: int, query_embedding: np.ndarray) -> list[schemas.ChunkRecord]:
    conn = create_connection()
    cursor = create_cursor_with_row_factory(conn)
    cursor.execute('''
    SELECT distance dist, c.rowid, c.chunk, r.rowid, r.title, r.url
    from vss_chunks v
    join chunks c on c.rowid = v.rowid 
    join resources r on r.id = c.resource_id
    where vss_search(
      v.chunk_embedding,
      vss_search_params(
        ?1,
        ?2
      )
    )
    order by dist
    ''', [query_embedding, n])
    results: list[schemas.ChunkRecord] = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def create_cursor_with_row_factory(conn):
    cursor = conn.cursor()
    cursor.row_factory = schemas.chunk_record_factory
    return cursor
