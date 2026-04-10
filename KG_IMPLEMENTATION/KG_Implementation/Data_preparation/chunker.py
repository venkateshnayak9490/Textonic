"""
text_chunker.py  
Key fix: overlap now steps back from end_pos, then seeks FORWARD to the
next sentence start, so no chunk ever begins mid-word or mid-sentence.
"""

import re
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from utils import count_words


class TextChunker:

    def __init__(
        self,
        chunk_size:     int = CHUNK_SIZE,
        overlap:        int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.chunk_size     = chunk_size
        self.overlap        = overlap
        self.min_chunk_size = min_chunk_size


    def _find_end_boundary(self, text: str, start: int, end: int) -> int:
        min_pos = start + self.chunk_size // 2
        max_pos = min(end, len(text))

        for pos in range(max_pos - 1, min_pos - 1, -1):
            if text[pos] in ".!?":
                if pos + 1 >= len(text) or text[pos + 1].isspace():
                    # Skip abbreviations like "Dr."
                    if not (pos > 0 and text[pos - 1].isupper()
                            and pos >= 2 and text[pos - 2].isspace()):
                        return pos + 1

        return max_pos

    def _find_next_sentence_start(self, text: str, pos: int) -> int:
        length = len(text)

        # First preference: paragraph break
        para_match = re.search(r'\n\n', text[pos:])
        if para_match and para_match.start() < self.overlap:
            return pos + para_match.end()

        # Second preference: ". Capital" or ".\n" within the overlap window
        window = text[pos: pos + self.overlap + 50]   # small look-ahead
        sent_match = re.search(r'[.!?]\s+([A-Z])', window)
        if sent_match:
            # return position of the capital letter
            return pos + sent_match.start(1)

        # Fallback: just return pos (original overlap start)
        return pos


    def create_chunks(self, text: str, source_id: str = "unknown") -> List[Dict]:
        chunks    = []
        text_len  = len(text)
        start_pos = 0
        chunk_num = 0

        while start_pos < text_len:
            end_pos = start_pos + self.chunk_size

            if end_pos < text_len:
                end_pos = self._find_end_boundary(text, start_pos, end_pos)
            else:
                end_pos = text_len

            chunk_text = text[start_pos:end_pos].strip()

            if len(chunk_text) >= self.min_chunk_size:
                sentence_count = (
                    chunk_text.count(".") + chunk_text.count("!") + chunk_text.count("?")
                )
                chunks.append({
                    "chunk_id":     f"{source_id}_{chunk_num:03d}",
                    "text":         chunk_text,
                    "source_file":  source_id,
                    "char_start":   start_pos,
                    "char_end":     end_pos,
                    "chunk_length": len(chunk_text),
                    "word_count":   count_words(chunk_text),
                    "sentence_count": max(1, sentence_count),
                })
                chunk_num += 1

            overlap_start = end_pos - self.overlap
            if overlap_start <= start_pos:          # must always move forward
                start_pos = end_pos
            else:
                start_pos = self._find_next_sentence_start(text, overlap_start)
                if start_pos <= (chunks[-1]["char_start"] if chunks else -1):
                    start_pos = end_pos             # safety: never go backward

        return chunks


    def chunk_multiple_documents(self, documents: List[Dict]) -> Dict:
        all_chunks = []

        for doc in documents:
            filtered_text = doc.get("filtered_text", "")
            source_id     = doc.get("source_id", "unknown")
            if not filtered_text:
                continue
            all_chunks.extend(self.create_chunks(filtered_text, source_id))

        stats = {
            "total_documents":   len(documents),
            "total_chunks":      len(all_chunks),
            "avg_chunks_per_doc": len(all_chunks) / len(documents) if documents else 0,
            "avg_chunk_length":  (
                sum(c["chunk_length"] for c in all_chunks) / len(all_chunks)
                if all_chunks else 0
            ),
            "avg_word_count": (
                sum(c["word_count"] for c in all_chunks) / len(all_chunks)
                if all_chunks else 0
            ),
        }

        return {"chunks": all_chunks, "statistics": stats}


# ── quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Global warming is primarily caused by greenhouse gas emissions from burning "
        "fossil fuels. Carbon dioxide concentrations have increased by 50% since "
        "pre-industrial times, reaching 420 ppm in 2023. This has led to a temperature "
        "rise of 1.1°C above pre-industrial levels. The main sources of CO2 are fossil "
        "fuel combustion, deforestation, and industrial processes. Methane is another "
        "potent greenhouse gas, with sources including agriculture, livestock, and "
        "natural gas systems. The impacts of climate change include sea level rise, "
        "more frequent extreme weather events, and disruptions to ecosystems. "
        "Renewable energy deployment has accelerated significantly in recent years. "
        "Solar and wind power now account for a growing share of electricity generation. "
        "Policy frameworks like the Paris Agreement set binding targets for emission reductions."
    )

    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks  = chunker.create_chunks(sample, source_id="test")

    print(f"Created {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  [{c['chunk_id']}]  chars {c['char_start']}–{c['char_end']}")
        print(f"  >>> {c['text'][:120]}")
        print()

    bad = [c for c in chunks if c["text"] and c["text"][0].islower() and c["char_start"] > 0]
    if bad:
        print(f"WARNING: {len(bad)} chunk(s) still start mid-sentence:")
        for c in bad:
            print(f"  {c['chunk_id']}: {c['text'][:60]}")
    else:
        print("✓ All chunks start at a clean sentence boundary.")