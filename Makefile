.PHONY: build test doc clean

build:
	mkdir -p build && cd build && cmake .. && make

test:
	mkdir -p build && cd build && cmake .. && make && make test

doc:
	doxygen
	rm -rf /tmp/libGPUGenie_doc
	cp -rf doc/html/ /tmp/libGPUGenie_doc/

clean:
	rm -rf build

