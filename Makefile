CXX = g++
CXXFLAGS = -g -Wall -Wextra -Icereal-1.2.2/include -pthread -std=c++0x -fopenmp -O3


all: bin bin/construct_index bin/query_change_point bin/centrality_single bin/centrality_multiple

bin:
	mkdir -p bin

bin/construct_index: sample/construct_index_main.cc src/historical_pruned_landmark_labeling_directed.cc
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^
	
bin/query_change_point: sample/query_change_point_main.cc src/historical_pruned_landmark_labeling_directed.cc
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^

bin/centrality_single: sample/centrality_single.cc src/historical_pruned_landmark_labeling_directed.cc
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^
	
bin/centrality_multiple: sample/centrality_multiple.cc src/historical_pruned_landmark_labeling_directed.cc
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^

.PHONY: test clean

test: bin bin/test
	bin/test

clean:
	rm -rf bin