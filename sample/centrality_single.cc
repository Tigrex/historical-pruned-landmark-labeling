#include "historical_pruned_landmark_labeling_directed.h"
#include <iostream>
#include <exception>
#include <chrono>
#include <cereal/archives/binary.hpp>
#include<cereal/types/vector.hpp>
using namespace std;

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: centrality_single file_name source_index" << endl;
    exit(EXIT_FAILURE);
  }

  historical_pruned_landmark_labeling hpll;

  cout << "File: " << argv[1] << endl;

  auto start = std::chrono::system_clock::now();
  ifstream ifs;
  ifs.open(argv[1], ios::binary);

  if (ifs.good()) {
    {
      cereal::BinaryInputArchive ia(ifs);
      try {
        ia >> hpll;
      }
      catch (std::exception const& e) {
        cout << e.what() << endl;
        ifs.close();
        exit(EXIT_FAILURE);
      }

    }

  }

  ifs.close();

  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  cout << "Load constructed indexes time: " << elapsed.count() << endl;

  
  int source = std::stoi(argv[2]);
  cout << "Source index: " << source << endl;
  start = std::chrono::system_clock::now();
  vector<double> centralities;
  hpll.get_centrality(source, centralities);
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  cout << "Query time: " << elapsed.count() << endl;

  for (double centrality: centralities) {
    cout << centrality << endl;
  }


}
