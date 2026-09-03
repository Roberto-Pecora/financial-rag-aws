"""Unit tests for section-aware chunking (SEC Part/Item boundary splitting).

Pure logic - no HTML fetch, no embedder, no Qdrant.
"""

from frag.sources.chunking import _section_split, _window_split


def test_splits_on_item_headers():
    text = (
        "Item 1. Business\n" + "a" * 100 + "\n"
        "Item 2. Properties\n" + "b" * 100 + "\n"
        "Item 7. MD&A\n" + "c" * 100
    )
    chunks = _section_split(text, chunk_size=1500, overlap=200)
    # Three item sections, each short enough to stay whole.
    assert len(chunks) == 3
    assert chunks[0].startswith("Item 1")
    assert chunks[1].startswith("Item 2")
    assert chunks[2].startswith("Item 7")


def test_short_section_kept_whole():
    text = "Item 7. Results\nRevenue increased 18% to $331,839 million."
    chunks = _section_split(text, chunk_size=1500, overlap=200)
    assert len(chunks) == 1
    assert "331,839" in chunks[0]


def test_long_section_falls_back_to_window():
    long_body = "x" * 5000
    text = "Item 7. MD&A\n" + long_body
    chunks = _section_split(text, chunk_size=1000, overlap=100)
    # A 5000+ char section must be window-split into multiple chunks.
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_preamble_before_first_header_is_kept():
    text = "Cover page text before any item.\nItem 1. Business\n" + "a" * 50
    chunks = _section_split(text, chunk_size=1500, overlap=200)
    assert any("Cover page" in c for c in chunks)


def test_no_headers_falls_back_to_window():
    text = "y" * 3000
    section = _section_split(text, chunk_size=1000, overlap=100)
    window = _window_split(text, chunk_size=1000, overlap=100)
    assert section == window


def test_part_headers_also_split():
    text = "PART I\n" + "a" * 50 + "\nPART II\n" + "b" * 50
    chunks = _section_split(text, chunk_size=1500, overlap=200)
    assert len(chunks) == 2
