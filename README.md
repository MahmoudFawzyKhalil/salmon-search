# Salmon Search
> Salmon Search is a handy Python CLI that indexes your articles and videos using semantic embeddings in sqlite to make it easy to search for and re-discover relevant articles based on their content.

```sh
# local dev
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# build
pip install build
python -m build

# test
python -m pytest
```