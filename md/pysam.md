# Summary of functions in pysam python package

## Retrieve query information

```{python3}
import pysam
s = pysam.AlignmentFile('example.bam')

query_with_soft_clipps = s.query_sequence
# Note: The sequence is returned as it is stored in the BAM file. 
# This will be the reverse complement of the original read sequence if the mapper has aligned the read to the reverse strand.

len(query_sequence) == s.query_length
query_len_with_soft_clipps = s.query_length

# This the index of the first base in query_sequence that is not soft-clipped
start_index_query = s.query_alignment_start
end_index_query = s.query_alignment_end
s.query_alignment_length == s.query_alignment_end - s.query_alignment_start
```

## Retrieve reference information
```{python}

```