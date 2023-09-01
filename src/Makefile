CC = g++ -Werror -Wextra -Wall -std=c++17
SRCS = model/*.cc
HEADERS = model/*.h
TESTS = tests.cc
LIB = network.a
PKG = `pkg-config --cflags --libs gtest`

all: install test

install:
	[ -d build ] || mkdir -p build
	cd build && qmake ../MLP.pro && make

uninstall:
	rm -rf build

network.a:
	$(CC) -c $(SRCS)
	ar rc $(LIB) *.o
	ranlib $(LIB)

style:
	cp ../materials/linters/.clang-format .clang-format
	clang-format -verbose -n *.cc model/*.cc model/*.h view/*.cc view/*.h controller/*.cc controller/*.h
	rm .clang-format

check: style leaks clean

dist:
	tar -cf MLP.tar *.cc *.h *.pro Makefile Doxyfile controller model view

clean:
	rm -rf *.o *.a *.dot *.gcno *.gcda *.info  gcovreport report build

rebuild: clean all
