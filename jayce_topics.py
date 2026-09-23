# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Download a broad topic catalog once, then use it offline."""

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CATALOG_PAGE = "Wikipedia:Vital articles/Level 4"
CATALOG_URL = "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level_4"


def _page_links(page):
    query = urlencode(dict(action="parse", format="json", formatversion=2,
                           page=page, prop="links", redirects=1, maxlag=5))
    request = Request("https://en.wikipedia.org/w/api.php?" + query, headers={
        "User-Agent": "Jayce/0.1 (https://github.com/Loophole-LLC/Jayce; topic catalog)",
    })
    with urlopen(request, timeout=15) as response:
        payload = response.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("topic response was too large")
    return json.loads(payload)["parse"]["links"]


def _validate_topics(topics):
    if not isinstance(topics, list) or not topics or any(
        not isinstance(topic, str) or not topic.strip() or len(topic) > 512 for topic in topics
    ):
        raise ValueError("invalid topic catalog")
    return sorted({topic.strip() for topic in topics})


def download_topics():
    # Discover the subject lists instead of hardcoding their names or scraping HTML.
    pages = sorted({link["title"] for link in _page_links(CATALOG_PAGE)
                    if link["ns"] == 4 and link["title"].startswith(CATALOG_PAGE + "/")})
    if not pages:
        raise ValueError("Wikipedia returned no topic lists")
    topics = []
    for page in pages:
        titles = [link["title"] for link in _page_links(page) if link["ns"] == 0]
        if not titles:
            raise ValueError(f"Wikipedia returned no topics for {page}")
        topics.extend(titles)
    return _validate_topics(topics)


def load_topics(path: Path, fallback):
    """Called under the trainer lock. A failed download never replaces the cache."""
    path = Path(path)
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache["source"] != CATALOG_URL:
            raise ValueError("unrecognized topic source")
        topics = _validate_topics(cache["topics"])
        print(f"Using {len(topics):,} cached Wikipedia topics.", flush=True)
        return topics
    except (OSError, ValueError, KeyError, TypeError):
        pass

    print("Downloading Wikipedia's topic catalog...", flush=True)
    try:
        topics = download_topics()
    except (OSError, ValueError, KeyError, TypeError) as error:
        topics = _validate_topics(list(fallback))
        print(f"Topic download failed: {error}. Using {len(topics):,} saved or starter topics.", flush=True)
        return topics

    temporary = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({"source": CATALOG_URL, "topics": topics},
                                        ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        print(f"Could not cache topics: {error}. Using the downloaded catalog this run.", flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Using {len(topics):,} Wikipedia topics.", flush=True)
    return topics
