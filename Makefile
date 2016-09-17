CC      = mpic++
CFLAGS  = -Wall -c --std=c++11 -O3
LDFLAGS = -Wall -O3 --std=c++11
ALL     = matrixmul
MATGENFILE = densematgen.o

all: $(ALL)

$(ALL): %: %.o $(MATGENFILE)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c matgen.h Makefile
	$(CC) $(CFLAGS) $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)

run: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1

runve0: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 0 -v

runve1: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1 -v

runve2: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 2 -v

runve3: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 3 -v

runvi: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1 -v -i

runvi8_2_0: all
	mpirun -np 8 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 0 -v -i

runvi8_2_1: all
	mpirun -np 8 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1 -v -i

runvi8_2_2: all
	mpirun -np 8 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 2 -v -i

runvi12_2: all
	mpirun -np 12 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1 -v -i

runvi9_3: all
	mpirun -np 9 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 3 -e 1 -v -i

runvige: all
	mpirun -np 6 ./matrixmul -f ../exported_tests/sparse05_00010_000 -s 3 -c 2 -e 1 -v -i -g 10