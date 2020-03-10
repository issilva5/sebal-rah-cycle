for i in $(seq 1 10)
do
	echo Executing ${i}th experiment
	echo Output repository created and starting execution
	time ./execute.sh /dev/shm/input /dev/shm/output 256 &&
	mv /dev/shm/output/*.csv /home/itallo/Downloads/Experimentos/CUDA/ExperimentoNovoInput/experimento${i}
        mv /dev/shm/output/*.txt /home/itallo/Downloads/Experimentos/CUDA/ExperimentoNovoInput/experimento${i}
        mv /dev/shm/output/ET24h.tif /home/itallo/Downloads/Experimentos/CUDA/ExperimentoNovoInput/experimento${i}
	rm /dev/shm/output/*
	echo Completing ${i}th experiment
done
