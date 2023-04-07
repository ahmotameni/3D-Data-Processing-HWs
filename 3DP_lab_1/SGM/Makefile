INC=`pkg-config --cflags --libs opencv4`
CXXFLAGS= -o3 -std=c++11

SOURCES=sgm.cpp main.cpp
EXECUTABLE=sgm

all:
	g++ $(CXXFLAGS)  $(SOURCES) -o $(EXECUTABLE) $(INC)
