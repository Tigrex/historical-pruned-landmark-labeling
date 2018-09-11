#include "historical_pruned_landmark_labeling_directed.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cereal/archives/binary.hpp>
using namespace std;

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: query GRAPH(input) INDEX(output)" << endl;
    exit(EXIT_FAILURE);
  }

  historical_pruned_landmark_labeling hpll;

  auto start = std::chrono::system_clock::now();
  hpll.construct_index(argv[1]);
  
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  cout << "Construct labels time: " << elapsed.count() << endl;

  ofstream ofs;
  ofs.open(argv[2], ios::binary);

  if (ofs.good()) {
    {
      cereal::BinaryOutputArchive oa(ofs);
      oa << hpll;
    }

  }

  ofs.close();
  cout << "Write labels to file" << endl;
  exit(EXIT_SUCCESS);

}