#include "historical_pruned_landmark_labeling_directed.h"
#include <iostream>
#include <exception>
#include <chrono>
#include <cereal/archives/binary.hpp>
#include<cereal/types/vector.hpp>
using namespace std;

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: centrality_test serialization_file_name source_id_file_name" << endl;
    exit(EXIT_FAILURE);
  }


  auto start = std::chrono::system_clock::now();
  historical_pruned_landmark_labeling hpll;
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
 


  vector<long> sourceIds;
  std::ifstream ifs2(argv[2]);
  for (long source_id; ifs2 >> source_id; ) {
    sourceIds.push_back(source_id);
  }
  cout << "Number of queries: " << sourceIds.size() << endl;


  start = std::chrono::system_clock::now();

  vector<double> centralities;
  for (int source: sourceIds) {
    hpll.get_centrality(source, centralities);
  }

  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  cout << "Total query time: " << elapsed.count() << endl;

}
