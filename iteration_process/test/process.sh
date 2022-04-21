sed -i 's/^/samtools faidx Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa /g' dna_loc
sh dna_loc >dna_seq
awk 'NR%8!=1' dna_seq >dna_seq_2
awk '{if(NR%7!=0)ORS=" ";else ORS="\n"}1' dna_seq_2 >dna_seq
sed -i s/[[:space:]]//g dna_seq
python encode.py
