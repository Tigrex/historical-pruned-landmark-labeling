// Copyright 2014, Takuya Akiba
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Takuya Akiba nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef HISTORICAL_PRUNED_LANDMARK_LABELING_H_
#define HISTORICAL_PRUNED_LANDMARK_LABELING_H_

#include <fstream>
#include<cereal/types/vector.hpp>


#define HPLL_CHECK(expr)                                                \
  if (expr) {                                                           \
  } else {                                                              \
    fprintf(stderr, "CHECK Failed (%s:%d): %s\n",                       \
            __FILE__, __LINE__, #expr);                                 \
    exit(EXIT_FAILURE);                                                 \
  }

class historical_pruned_landmark_labeling {
 public:
  struct label_entry_t {
    int v, d, t;
   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
      ar & v;
      ar & d;
      ar & t;
    }
  };

  historical_pruned_landmark_labeling() : V(0), T(0){}

  // Constructs an index from a graph, given as a list of edges.
  // Each edge is described as a tuple of creation time and vertices (in this order).
  // Vertices should be described by numbers starting from zero.
  // Time should be described by non-negative integers.
  void construct_index(const char *filename);
  void construct_index(std::istream &ifs);
  void construct_index(const std::vector<std::tuple<int, int, int>> &es);

  // Index access
  void get_label(int v, std::vector<label_entry_t> &f_label, std::vector<label_entry_t> &r_label);
  double get_average_forward_label_size();  // average number of entries
  double get_average_reverse_label_size();  // average number of entries



  // Returns the distance between vertices |v| and |w| at time |t|.
  int query_snapshot(int v, int w, int t);

  // cp = [(t_0, d_0), (t_1, d_1), ...], where the distance between |v| and |w|
  // gets |d_i| for the first time at time |t_i|.
  // (i.e., t_0 < t_1 < ..., and d_0 > d_1 > ...)
  void query_change_points(int v, int w, std::vector<std::pair<int, int>> &cp);

  void get_centrality(int v, std::vector<double> &result);


 private:
  struct edge_t {
    int v, t;
   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
      ar & v;
      ar & t;
    }
  };

  struct label_entry_cmp {
    bool operator()(const label_entry_t &a, const label_entry_t &b) const {
      if (a.v != b.v) return a.v < b.v;
      else return a.d < b.d;
    }
  };

  // Graph and index
  int V;
  int T;
  std::vector<std::vector<label_entry_t>> f_labels;
  std::vector<std::vector<label_entry_t>> r_labels;
  std::vector<std::vector<edge_t>> f_adj;
  std::vector<std::vector<edge_t>> r_adj;
  std::vector<int> ord;


  void get_root_order(std::vector<int> &ord);
  
  friend class cereal::access;
  template<class Archive>
  void serialize(Archive & ar) {
    ar & V;
    ar & T;
    ar & f_labels;
    ar & r_labels;
  }

};

#endif  // HISTORICAL_PRUNED_LANDMARK_LABELING_H_