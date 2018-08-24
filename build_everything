EIGEN=/usr/include/eigen3



cd ./bin
for file in ../src/*cpp; do
	echo compiling $file;
	g++ -c -I$EIGEN/include -I../include -O3 $file; 
done
ar crv libpf.a *.o
cd ..


