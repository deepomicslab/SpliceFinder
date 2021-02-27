import random
import linecache
import numpy as np


gene_num = 5237



# random choose a gene(left-1000,right+1000)
a = random.randint(1,gene_num)
gene = linecache.getline(r'gene_loc_selected', a)

chrom = gene.split('\t')[0]
start = int(gene.split('\t')[1]) - 1000
stop = int(gene.split('\t')[2]) + 1000


length = stop -start + 1
sample_num = length - 399


print("chromosome is %s"%chrom)
print("from %s"%start)
print("to %s"%stop)

# find the splice site on the sequence
left_splice = []
right_splice = []

exon_address = 'chromosome/'+chrom+'/exon_uniq'
left_address = 'chromosome/'+chrom+'/left_uniq'
right_address = 'chromosome/'+chrom+'/right_uniq'

print("splice site:")
with open (exon_address) as input:
    for line in input:
        if int(line) >= start and int(line) <=stop:
            print(line)
            left = open(left_address,'r')
            right = open(right_address,'r')
            for left_loc in left:
                if int(left_loc)==int(line):
                    print('left')
                    if(int(line) not in left_splice):
                        left_splice.append(int(line))
                    break
            for right_loc in right:
                if int(right_loc)==int(line):
                    print('right')
                    if(int(line) not in right_splice):
                        right_splice.append(int(line))
            left.close()
            right.close()

print(left_splice)
print(right_splice)

# generate locations

with open ('dna_loc','w') as output:
    for i in range(start,start+sample_num):
        a=chrom
        b=i
        c=i+399
        output.write(str(a)+":"+str(b)+"-"+str(c)+"\n")

# generate labels

label=[]
label2=[]

tran_left=np.array(np.loadtxt('transcript_left'))
tran_right=np.array(np.loadtxt('transcript_right'))



for i in range(sample_num):
    label.append(2)
    label2.append(2)

for splice in left_splice:
    x = splice-start-200
    if (splice not in tran_left):
        label2[x]=0


for splice in right_splice:
    x = splice-start-199
    if (splice not in tran_right):
        label2[x]=1




label2_file = open('y_label','w')
for i in range(sample_num):
    label2_file.write(str(label2[i]))
    label2_file.write('\t')

label2_file.close()







