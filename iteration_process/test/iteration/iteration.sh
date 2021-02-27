for((i=1;i<=100;i++));
do   
        cd ../../generate_seq
        python generate.py
        mv dna_loc y_label ../test
        cd ../test
        sh process.sh
	mv encoded_seq y_label iteration/
        cd iteration/
        python test_Cla.py
        cmd="mv CNN.h5 CNN_${i}.h5"
	$cmd
        cp fp_seq train/
        cd train/
        cat fp_seq >>encoded_seq.txt
        
        cmd="cp encoded_seq.txt encoded_seq_${i}.txt"
	$cmd
	
	count=`cat fp_seq |wc -l`
	for((j=1;j<=$[count];j++));
	do
		echo "2" >>label.txt

        done
        
        python CNN.py
        mv CNN.h5 ../
	cd ..
	cmd="mkdir iter_${i}"
	$cmd
        cmd="mv y_label fp_seq iter_${i}"
	$cmd

done
