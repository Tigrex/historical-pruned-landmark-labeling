#include "historical_pruned_landmark_labeling_directed.h"
#include <iostream>
#include <cereal/archives/binary.hpp>
using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "usage: query_change_point INDEX" << endl;
    exit(EXIT_FAILURE);
  }

  historical_pruned_landmark_labeling hpll;

  ifstream ifs;
  ifs.open(argv[1], ios::binary);

  HPLL_CHECK(ifs);
  cereal::BinaryInputArchive ia(ifs);
  ia >> hpll;

  for (int u, v; cin >> u >> v; ) {
    vector<pair<int, int>> cp;
    hpll.query_change_points(u, v, cp);
    for (auto p : cp) {
      cout << p.first << ":" << p.second << "\t";
    }
    cout << endl;
  }
}