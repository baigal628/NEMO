samtools view -b -s 0.001 220308_ang_0.sorted.bam -o /private/groups/brookslab/gabai/tools/NEMO/test/input/ang_0.sorted.bam
samtools view -b -s 0.001 220308_ang_500.sorted.bam -o /private/groups/brookslab/gabai/tools/NEMO/test/input/ang_500.sorted.bam
samtools view -b -s 0.001 20210615_0802_chrom.sorted.bam -o /private/groups/brookslab/gabai/tools/NEMO/test/input/chrom_ang_500.sorted.bam

for i in *.sorted.bam; do samtools index $i ;done

python3 ../create_test_data.py ang_0.sorted.bam
python3 ../create_test_data.py ang_500.sorted.bam
python3 ../create_test_data.py chrom_ang_500.sorted.bam

pod5 subset --missing-ok --csv ang_0.sorted_target_mapping.csv /private/groups/brookslab/data.rep/addseq_submission/revision/220308_ang_0.pod5
pod5 subset --missing-ok --csv ang_500_target_mapping.csv /private/groups/brookslab/data.rep/addseq_submission/revision/220308_ang_500.pod5
pod5 subset --missing-ok --csv chrom_ang_500_target_mapping.csv /private/groups/brookslab/data.rep/addseq_submission/revision/20210615_0802_chrom.pod5