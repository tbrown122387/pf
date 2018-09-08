EIGEN=/usr/include/eigen3



cd ./bin
for file in ../src/*cpp; do
	echo g++ -c -I$EIGEN -I../include -O3 $file;
	g++ -std=c++11 -fPIC -c -I$EIGEN -I../include -O3 $file; 
done
ar crv libpf.a *.o
cd ..


