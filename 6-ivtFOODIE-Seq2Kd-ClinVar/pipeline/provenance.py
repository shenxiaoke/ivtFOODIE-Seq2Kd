"""Complete source file inventory for the frozen run; generated inputs have separate task hashes."""
from common import *
def write_input_manifest():
 sources=[SHARE/'clinvar/clinvar_20250623.vcf',SHARE/'clinvar/hg38.fa',SHARE/'clinvar/hg38.fa.fai',SHARE/'clinvar/gencode.v48.basic.annotation.gtf',SHARE/'clinvar/human_tf_final2_1636.csv',TFDIR/'human_tf_1636_full_seq.csv']
 sources += [TFDIR/'junchen_new_20250426'/f for f in ['human_tf_part1_1545.csv','human_tf_part2_concat.csv','human_tf_part2_split.csv','human_tf_part3_83.csv']]
 sources += sorted((ARCHIVE/'dna_mutation_sk/Seq2Kd_motif').glob('streme_out_100.part[12]/*/streme.txt'))
 save_json([dict(path=str(f),resolved=str(f.resolve()),size=f.stat().st_size,sha256=filehash(f)) for f in sources],RUN/'qc/input_manifest.json')
 return len(sources)
if __name__=='__main__':print('Source files hashed:',write_input_manifest(),flush=True)
