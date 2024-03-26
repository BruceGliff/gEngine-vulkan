#include "gEng/utils/Formatter.hpp"

#include <gtest/gtest.h>

#include <sstream>

using namespace gEng;

namespace {

TEST(FormatterTest, SimpleString) {
  std::stringstream OS;

  static constexpr auto Answer = "Simple string";
  out(OS) << Answer;

  EXPECT_EQ(OS.str(), std::string(Answer));
}

TEST(FormatterTest, SimpleString1) {
  std::stringstream OS;

  out(OS) << "Simple string";

  EXPECT_EQ(OS.str(), std::string("Simple string"));
}

TEST(FormatterTest, TwoStrings) {
  std::stringstream OS;

  static constexpr auto Answer = "Simple string";
  out(OS) << "Simple string" << '+' << Answer;

  EXPECT_EQ(OS.str(), std::string("Simple string") + '+' + Answer);
}

struct HasPrint final {
  static constexpr auto Name = "HasPrint";
  void print(std::ostream &OS) const { OS << Name; }
};
struct HasOutputOperator final {
  static constexpr auto Name = "HasOutputOperator";
  friend std::ostream &operator<<(std::ostream &OS, HasOutputOperator) {
    OS << Name;
    return OS;
  }
};

struct NoCopy final {
  NoCopy() = default;
  NoCopy(NoCopy const &) = delete;
  NoCopy(NoCopy &&) = default;
  NoCopy &operator=(NoCopy const &) = delete;
  NoCopy &operator=(NoCopy &&) = default;

  static constexpr auto ForLambda = "LambdaNoCopy";
  static constexpr auto Static = "StaticNoCopy";
};

static void printNoCopy(const NoCopy &, std::ostream &OS) {
  OS << NoCopy::Static;
}

TEST(FormatterTest, CustomClasses) {
  std::stringstream OS;

  HasOutputOperator LVal;
  static constexpr auto String = "and string";
  out(OS) << HasPrint{} << '_' << LVal << '_' << String;

  EXPECT_EQ(OS.str(), std::string() + HasPrint::Name + '_' +
                          HasOutputOperator::Name + '_' + String);
}

TEST(FormatterTest, NoCopy) {
  std::stringstream OS;

  NoCopy Val;
  out(OS, printNoCopy) << Val;
  out(OS, [](const NoCopy &, std::ostream &OS) { OS << NoCopy::ForLambda; })
      << NoCopy{};

  EXPECT_EQ(OS.str(), std::string() + NoCopy::Static + NoCopy::ForLambda);
}

struct NoDefPrint final {
  static constexpr auto Name = "NoDefPrint";
};

TEST(FormatterTest, NoDefPrint) {
  std::stringstream OS;

  NoCopy Val;
  auto Printer = [](const NoDefPrint &, std::ostream &OS) {
    OS << NoDefPrint::Name;
  };
  out(OS, printNoCopy, Printer) << Val << NoDefPrint{};

  EXPECT_EQ(OS.str(), std::string() + NoCopy::Static + NoDefPrint::Name);
}

TEST(FormatterTest, Array) {
  std::stringstream OS;

  auto Arr = std::array{NoCopy{}, NoCopy{}, NoCopy{}};
  auto Printer = [](const NoDefPrint &, std::ostream &OS) {
    OS << NoDefPrint::Name;
  };

  out(OS, printNoCopy, Printer) << Arr << NoDefPrint{};

  std::string Answer{};
  for (auto &&X : Arr) {
    Answer += X.Static;
    Answer += ' ';
  }
  Answer.erase(Answer.size() - 1);

  EXPECT_EQ(OS.str(), Answer + NoDefPrint::Name);
}

TEST(FormatterTest, Optional) {
  std::stringstream OS;

  std::optional<NoCopy> Opt{NoCopy{}};
  std::optional<NoCopy> NoOpt{};
  out(OS, printNoCopy) << Opt << NoOpt;

  EXPECT_EQ(OS.str(), std::string() + NoCopy::Static + "None");
}

TEST(FormatterTest, Pointer) {
  std::stringstream OS;

  std::unique_ptr<NoCopy> Obj = std::make_unique<NoCopy>();
  auto *Ptr = Obj.get();

  out(OS, printNoCopy) << Ptr << (NoCopy *)nullptr;

  EXPECT_EQ(OS.str(), std::string() + NoCopy::Static + "Null");
}

TEST(FormatterTest, ArrayOfArrayOfOpionalOfPointer) {
  std::stringstream OS;

  NoDefPrint Obj{};
  NoDefPrint *Ptr = &Obj;

  auto Arr = std::array{std::array{std::optional<NoDefPrint *>{Ptr},
                                   std::optional<NoDefPrint *>{std::nullopt}},
                        std::array{std::optional<NoDefPrint *>{nullptr},
                                   std::optional<NoDefPrint *>{Ptr}}};

  out(OS, [](const NoDefPrint &, std::ostream &OS) { OS << NoDefPrint::Name; })
      << Arr;

  EXPECT_EQ(OS.str(), std::string() + NoDefPrint::Name + ' ' + "None" + ' ' +
                          "Null" + ' ' + NoDefPrint::Name);
}

TEST(FormatterTest, NoCopyLambda) {

  struct Lambda {
    void operator()(NoDefPrint, std::ostream &OS) const {
      OS << NoDefPrint::Name;
    }

    Lambda() = default;
    Lambda(Lambda const &) = delete;
    Lambda(Lambda &&) = default;

    Lambda &operator=(Lambda const &) = delete;
    Lambda &operator=(Lambda &&) = default;
  };

  std::stringstream OS;

  out(OS, Lambda{}) << NoDefPrint{};

  EXPECT_EQ(OS.str(), std::string() + NoDefPrint::Name);
}

TEST(FormatterTest, NoMoveLambda) {

  struct Lambda {
    void operator()(NoDefPrint, std::ostream &OS) const {
      OS << NoDefPrint::Name;
    }

    Lambda() = default;
    Lambda(Lambda const &) = delete;
    Lambda(Lambda &&) = delete;

    Lambda &operator=(Lambda const &) = delete;
    Lambda &operator=(Lambda &&) = delete;
  };

  std::stringstream OS;

  Lambda L{};
  out(OS, L) << NoDefPrint{};

  EXPECT_EQ(OS.str(), std::string() + NoDefPrint::Name);
}

} // namespace
