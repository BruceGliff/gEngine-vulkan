#include <gtest/gtest.h>

namespace {

TEST(Simple, OneEqOne) {
  EXPECT_EQ(1, 1);
  EXPECT_NE(1, 0);
}

} // namespace
