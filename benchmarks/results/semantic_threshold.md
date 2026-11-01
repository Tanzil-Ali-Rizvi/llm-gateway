# Semantic cache: hit rate vs wrong answers (BAAI/bge-small-en-v1.5, 4,000 QQP pairs)

| Threshold | Duplicates served from cache | Different questions wrongly served | Precision of hits |
|---|---|---|---|
| 0.80 | 94.9% | 36.15% | 61.8% |
| 0.85 | 80.9% | 22.62% | 68.8% |
| 0.88 | 68.2% | 14.66% | 74.1% |
| 0.90 | 58.6% | 10.82% | 76.9% |
| 0.92 | 46.3% | 6.58% | 81.2% |
| 0.94 | 34.2% | 3.23% | 86.7% |
| 0.96 | 21.9% | 1.33% | 91.0% |
| 0.98 | 12.4% | 0.32% | 95.9% |

Most similar *non-duplicate* pairs (what a too-low threshold would get wrong):

- 0.997: "What are some accessories that would be perfect with this latex maids outfit?" vs "What accessories would be perfect with this latex maids outfit?"
- 0.997: "Which is the best institute for Hadoop and gives placements in Bangalore?" vs "Which is the best Institute for Hadoop and which gives good placements in Bangalore?"
- 0.995: "Which is the best institute for coaching of mass communication in delhi ?" vs "Which is the best coaching institute for mass communication in delhi?"
- 0.993: "How can I solve this AP computer science question?" vs "How would you solve this AP Computer Science question?"
- 0.985: "How do i make videos like this ?" vs "How can I make a video like this?"
