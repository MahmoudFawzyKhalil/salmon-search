from pathlib import Path

import typer

from .schemas import Resource, ChunkRecord

app = typer.Typer(rich_markup_mode="rich")


@app.command()
def init():
    """
    Initialize the sqlite database. Must be called before using any other command.
    """
    from . import db
    import click

    try:
        db.create_db()
    except db.DbAlreadyExistsException:
        raise click.ClickException("Database already exists.")


@app.command()
def index(url: str = typer.Option(None, help="URL of the resource to index."),
          file: Path = typer.Option(None, help="Path to a file with URLs of the resources to index.")):
    """
    Index a resource e.g. an article, or a YouTube video.

    salmon index --url https://en.wikipedia.org/wiki/Salmon
    salmon index --file urls.txt
    """
    import click

    from . import db
    db.update_vss_index()

    if url is not None:
        resources = [index_url(url)]
    elif file is not None:
        resources = index_file(file)
    else:
        raise click.ClickException("Either --url or --file must be specified.")
    db.update_vss_index()
    print_resource_table(resources)


def index_file(file: Path) -> list[Resource]:
    resources: list[Resource] = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                resources.append(index_url(line))
    return resources


def index_url(url: str) -> Resource | None:
    from . import db
    from . import resources
    import rich

    valid = validate_url(url)
    if not valid:
        return None
    try:
        resource: Resource = resources.create_resource(url)
        db.save_resource(resource)
    except Exception as e:
        rich.print(f"[red]Error saving resource {url} to database. Skipping. Error:{str(e)}[/red]")
        return None
    return resource


def print_resource_table(resources: list[Resource]):
    import rich.table as table
    import rich

    console = rich.console.Console()

    tbl: table.Table = table.Table("id",
                                   table.Column("url"),
                                   "title",
                                   "#chunks",
                                   title="Resources")
    for resource in [r for r in resources if r is not None]:
        tbl.add_row(str(resource.id), resource.url, resource.title, str(len(resource.chunks)))

    console.print(tbl)


def print_matches_table(matches: list[ChunkRecord]):
    import rich.table as table
    import rich

    console = rich.console.Console()

    tbl: table.Table = table.Table("#",
                                   "dist",
                                   "rid",
                                   table.Column("resource_title", max_width=40),
                                   "resource_url",
                                   # "most_similar_chunk",
                                   "cid",
                                   title="Matches")
    for i, chunk in enumerate(matches):
        tbl.add_row(
            str(i),
            f"{chunk.distance:.2f}",
            str(chunk.resource_id),
            chunk.resource_title,
            chunk.resource_url,
            # chunk.chunk,
            str(chunk.chunk_id)
        )

    console.print(tbl)


@app.command()
def search(query: str = typer.Argument(..., help="Search query."),
           n: int = typer.Option(100, help="Number of text chunks to compare the query to."),
           o: str = typer.Option("table", help="Output format. One of: table, json.")):
    """
    Semantically search resources.

    salmon search "Where do Salmons live?"
    """
    import rich
    from . import db
    from . import embeddings

    query_embedding = embeddings.encode(query)
    results: list[ChunkRecord] = db.get_most_similar_articles_based_on_n_chunks(n, query_embedding)
    if o == "table":
        print_matches_table(results)
    elif o == "json":
        rich.print_json(data=results)


@app.command()
def delete(resource_id: int = typer.Argument(..., help="resource_id for resource to delete and un-index.")):
    """
    Delete a resource from the index.

    salmon delete 1
    """
    import rich
    from . import db

    resource = db.delete_resource(resource_id)
    print_resource_table([resource])

@app.command()
def get(rid: int = typer.Option(None, help="Resource ID of resource to get."),
        cid: int = typer.Option(None, help="Chunk ID of chunk to get")):
    """
    Get a resource or chunk.

    salmon get --rid 1
    """
    import click
    import rich

    from . import db

    if rid is not None:
        resource = db.get_resource(rid)
        print_resource_table([resource])
    elif cid is not None:
        chunk = db.get_chunk(cid)
        rich.print(chunk)
    else:
        raise click.ClickException("Either --rid or --cid must be specified.")


def validate_url(url) -> bool:
    import validators
    from . import db
    import rich

    if not validators.url(url):
        rich.print(f"[red]Invalid URL: {url}, skipping.[/red]")
        return False
    if db.resource_exists_by_url(url):
        # rich.print(f"[green]Resource already exists: {url}, skipping.[/green]")
        return False
    return True


if __name__ == "__main__":
    app()
