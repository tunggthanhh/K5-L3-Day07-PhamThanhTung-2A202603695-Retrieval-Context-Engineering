# Day 6 imports

This directory imports the reusable, permitted corpus inputs from Day 6:

- `bo_luat_lao_dong.pdf`: Bộ luật Lao động số 45/2019/QH14.
- `manifest.json`: source, access, page-count, and permission record.
- `questions.json`: the original 20 answerable Day 6 benchmark cases (15 dev, 5 test).

The Day 6 questions retain their original gold fields for offline evaluation only.
They must never be passed to the retrieval tools or language model.  They are also
insufficient for Day 7 by themselves: add at least two unanswerable cases to each
split before running the Day 7 benchmark.

`src/day6/` contains the reused ingestion, chunking, embedding, local-Qdrant store,
and evaluation primitives.  The old Qdrant index and historical benchmark results
were intentionally not imported: rebuild and measure a fresh Day 7 index.
