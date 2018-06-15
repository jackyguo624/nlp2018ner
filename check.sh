for i in `seq 0 9`; do
	echo "$1/epoch0${i}_stat.txt"
	./conlleval < $1/epoch0${i}_stat.txt
done

for i in `seq 10 24`; do
	echo "$1/epoch${i}_stat.txt"
	./conlleval < $1/epoch${i}_stat.txt
done