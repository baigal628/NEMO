python3  ../../src/analyzeKmer.py \
    --posbam /private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/pod5/220517_ang_500.sorted.bam \
    --negbam /private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/pod5/220308_ang_0.sorted.bam \
    --posparq /private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/pod5/220517_ang_500_kmer-kmersignal-complete.parquet \
    --negparq /private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/pod5/220308_ang_0_kmer_new-kmersignal-complete.parquet \
    --outpath /private/groups/brookslab/gabai/projects/Add-seq/results/figures/ \
    --prefix 240908_sphe_pos_nuclei_neg \
    --max_batch 1 \
