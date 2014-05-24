#include "../lib/gtest/gtest.h"
#include "historical_pruned_landmark_labeling.h"
#include <climits>
using namespace std;

TEST(hpll, triangle) {
  vector<tuple<int, int, int>> es ={
    make_tuple(1, 0, 1),
    make_tuple(2, 1, 2),
    make_tuple(3, 0, 2),
  };

  historical_pruned_landmark_labeling hpll;
  hpll.construct_index(es);

  ASSERT_EQ(hpll.query_snapshot(0, 1, 0), -1);
  ASSERT_EQ(hpll.query_snapshot(0, 2, 0), -1);
  ASSERT_EQ(hpll.query_snapshot(1, 2, 0), -1);

  ASSERT_EQ(hpll.query_snapshot(0, 1, 1), 1);
  ASSERT_EQ(hpll.query_snapshot(0, 2, 1), -1);
  ASSERT_EQ(hpll.query_snapshot(1, 2, 1), -1);

  ASSERT_EQ(hpll.query_snapshot(0, 1, 2), 1);
  ASSERT_EQ(hpll.query_snapshot(0, 2, 2), 2);
  ASSERT_EQ(hpll.query_snapshot(1, 2, 2), 1);

  ASSERT_EQ(hpll.query_snapshot(0, 1, 3), 1);
  ASSERT_EQ(hpll.query_snapshot(0, 2, 3), 1);
  ASSERT_EQ(hpll.query_snapshot(1, 2, 3), 1);

  ASSERT_EQ(hpll.query_snapshot(0, 1, 123123123), 1);
  ASSERT_EQ(hpll.query_snapshot(0, 2, 123123123), 1);
  ASSERT_EQ(hpll.query_snapshot(1, 2, 123123123), 1);

  ASSERT_EQ(hpll.query_snapshot(0, 123123123, 0), -1);
  ASSERT_EQ(hpll.query_snapshot(-1, 0, 0), -1);

  {
    vector<pair<int, int>> cp, ans =
      {make_pair(0, -1), make_pair(2, 2), make_pair(3, 1)};
    hpll.query_change_points(0, 2, cp);
    ASSERT_EQ(cp, ans);
  }
}
