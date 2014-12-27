.PHONY: build test doc clean

build:
	mkdir -p build && cd build && cmake .. && make

test:
	mkdir -p build && cd build && cmake .. && make && make test

doc:
	doxygen

clean:
	rm -rf build