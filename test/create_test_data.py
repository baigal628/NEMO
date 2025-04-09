import pysam
import sys
def create_pod5_csv(bam):
    out_csv = open(bam.split(".sorted.bam")[0] + "_target_mapping.csv", 'w')
    out_csv.write("target,read_id\n")
    target = bam.split(".sorted.bam")[0] + "_downsampled.pod5"
    with pysam.AlignmentFile(bam, "rb") as samfile:
        for s in samfile:
            if s.is_mapped and not s.is_secondary and not s.is_supplementary:
                read_id = s.query_name
                out_csv.write(f"{target},{read_id}\n")
    out_csv.close()
if __name__ == "__main__":
    create_pod5_csv(sys.argv[1])