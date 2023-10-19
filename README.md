# seqUtils
A deep learning model to predict modifications on long read data

## Usage
```{python}
import sys
from seqUtil import *
from bamUtil import *
from nanoUtil import *
from nntUtil import *
from modPredict import *
```

```{python}
neg_bam = './projects/Add-seq/data/ctrl/mapping/unique.0.pass.sorted.bam'
ref = './projects/Add-seq/data/ref/sacCer3.fa'
neg_evt = './projects/Add-seq/data/ctrl/eventalign/unique.0.eventalign.tsv'

modPredict(bam = neg_bam, event = neg_evt, region = 'PHO5', genome=ref, prefix = 'PHO5_neg')
```