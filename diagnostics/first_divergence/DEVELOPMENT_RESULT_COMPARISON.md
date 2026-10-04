# Development-result comparison

The older development results below are retained for traceability. They do not
describe the selected deployment. The reported results correspond to the
seed-42 software reference and the FPGA captures included in this repository.

| Quantity | Earlier development result | Reported result |
|---|---:|---:|
| Selected Python true-INT8 correct predictions | 2536/3200 | 2488/3200 |
| Selected Python true-INT8 accuracy | 79.25% | 77.75% |
| FPGA correct predictions | 2535/3200 | 2497/3200 |
| FPGA accuracy | 79.22% | 78.03125% (78.03% rounded) |
| FPGA/Python-primary matches | 3106/3200 | 3006/3200 |
| FPGA/Python-primary agreement | 97.06% | 93.9375% (93.94% rounded) |
| FPGA/Python-primary mismatches | 94/3200 | 194/3200 |
| Board500 FPGA correct predictions | 411/500 | 404/500 |
| Board500 FPGA accuracy | 82.2% | 80.8% |
| Board500 FPGA/Python-primary matches | 493/500 | 473/500 |
| Board500 FPGA/Python-primary agreement | 98.6% | 94.6% |

The reported full-test confusion matrix, computed from the 3,200 UART `PRED`
records, is:

```text
                 predicted
true          airplane  car  dog  ship
airplane           677   33   35    55
car                112  559   48    81
dog                 34   10  745    11
ship               217   36   31   516
```

The corresponding current frozen software/HLS distinction is:

- Python true-INT8 accuracy: 2488/3200 = 77.75%;
- HLS CNN-only C-simulation accuracy: 2488/3200 = 77.75%;
- Python/HLS final-prediction agreement: 3200/3200 = 100%.

The exact hardware-equivalent replay and physical checkpoint comparisons are
documented in this directory. They are separate from the software-reference
CNN-only C-simulation result.
