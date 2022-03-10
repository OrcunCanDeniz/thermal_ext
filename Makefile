all: nms/nms.hpp
	g++  roi_select.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -o roi -g

run:
	./mser

clean:
	rm -rf mser