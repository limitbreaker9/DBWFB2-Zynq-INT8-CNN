# Controlled boundary-only comparison

These retained results use the frozen selected DBWFB2 seed-42 deployment with
gain 1, `C1_SHIFT=9` and `C2_SHIFT=8`. The same CNN, deployed integer arithmetic
and implemented stream convention, including retained transaction state, were
held fixed. Only trailing replicate versus zero extension changed.

| Trailing extension | Correct / test images | Accuracy |
|---|---:|---:|
| Replicate | 2,497/3,200 | 78.03% |
| Zero | 2,497/3,200 | 78.03% |

Predicted labels differ for 55/3,200 images. The boundary rule changes individual
predictions but does not change overall accuracy in this frozen comparison.
No accuracy advantage is claimed for replicate extension. This comparison is
separate from the five-seed software evaluation.

## Saved result records

- [boundary_ablation_metrics.csv](boundary_ablation_metrics.csv): accuracy, macro-F1 and clipping counts for both settings.
- [boundary_ablation_summary.json](boundary_ablation_summary.json): aggregate results, prediction agreement and tensor differences.
- [changed_prediction_indices.csv](changed_prediction_indices.csv): the 55 changed indices, true labels and predictions.

These are saved result records. No dedicated boundary-ablation runner is included.
