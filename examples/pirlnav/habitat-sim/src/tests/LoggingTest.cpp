// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "esp/core/Logging.h"

#include <sstream>

#include <Corrade/Containers/StaticArray.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringStl.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/TestSuite/Tester.h>

namespace Cr = Corrade;

namespace esp {

namespace sim {
namespace test {
namespace {
void debug(const Cr::Containers::StringView statement) {
  ESP_DEBUG() << statement;
}
void warning(const Cr::Containers::StringView statement) {
  ESP_WARNING() << statement;
}
}  // namespace
}  // namespace test
}  // namespace sim

namespace gfx {
namespace test {
namespace {
void debug(const Cr::Containers::StringView statement) {
  ESP_DEBUG() << statement;
}
void warning(const Cr::Containers::StringView statement) {
  ESP_WARNING() << statement;
}
}  // namespace
}  // namespace test
}  // namespace gfx
}  // namespace esp
namespace {

struct LoggingTest : Cr::TestSuite::Tester {
  explicit LoggingTest();

  void envVarTest();
};

constexpr const struct {
  const char* envString;
  const char* defaultDebug;
  const char* defaultWarning;
  const char* simDebug;
  const char* simWarning;
  const char* gfxDebug;
  const char* gfxWarning;
} EnvVarTestData[]{
    {nullptr, "[Default] LoggingTest.cpp(103)::envVarTest : DebugDefault\n",
     "[Default] LoggingTest.cpp(107)::envVarTest : WarningDefault\n",
     "[Sim] LoggingTest.cpp(23)::debug : DebugSim\n",
     "[Sim] LoggingTest.cpp(26)::warning : WarningSim\n",
     "[Gfx] LoggingTest.cpp(36)::debug : DebugGfx\n",
     "[Gfx] LoggingTest.cpp(39)::warning : WarningGfx\n"},
    {"debug", "[Default] LoggingTest.cpp(103)::envVarTest : DebugDefault\n",
     "[Default] LoggingTest.cpp(107)::envVarTest : WarningDefault\n",
     "[Sim] LoggingTest.cpp(23)::debug : DebugSim\n",
     "[Sim] LoggingTest.cpp(26)::warning : WarningSim\n",
     "[Gfx] LoggingTest.cpp(36)::debug : DebugGfx\n",
     "[Gfx] LoggingTest.cpp(39)::warning : WarningGfx\n"},
    {"quiet", "", "", "", "", "", ""},
    {"error", "", "", "", "", "", ""},
    {"quiet:Sim,Gfx=verbose", "", "",
     "[Sim] LoggingTest.cpp(23)::debug : DebugSim\n",
     "[Sim] LoggingTest.cpp(26)::warning : WarningSim\n",
     "[Gfx] LoggingTest.cpp(36)::debug : DebugGfx\n",
     "[Gfx] LoggingTest.cpp(39)::warning : WarningGfx\n"},
    {"warning:Gfx=debug", "",
     "[Default] LoggingTest.cpp(107)::envVarTest : WarningDefault\n", "",
     "[Sim] LoggingTest.cpp(26)::warning : WarningSim\n",
     "[Gfx] LoggingTest.cpp(36)::debug : DebugGfx\n",
     "[Gfx] LoggingTest.cpp(39)::warning : WarningGfx\n"},
};  // EnvVarTestData

LoggingTest::LoggingTest() {
  addInstancedTests({&LoggingTest::envVarTest},
                    Cr::Containers::arraySize(EnvVarTestData));
}

void LoggingTest::envVarTest() {
  auto&& data = EnvVarTestData[testCaseInstanceId()];

  esp::logging::LoggingContext ctx{data.envString};

  std::ostringstream out;
  Cr::Utility::Debug debugCapture{&out};
  Cr::Utility::Warning warnCapture{&out};
  // use contains to bypass issue with timestamp

  ESP_DEBUG() << "DebugDefault";
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.defaultDebug}));
  out.str("");
  ESP_WARNING() << "WarningDefault";
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.defaultWarning}));
  out.str("");
  esp::sim::test::debug("DebugSim");
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.simDebug}));
  out.str("");
  esp::sim::test::warning("WarningSim");
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.simWarning}));
  out.str("");

  esp::gfx::test::debug("DebugGfx");
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.gfxDebug}));
  out.str("");
  esp::gfx::test::warning("WarningGfx");
  CORRADE_VERIFY(Cr::Containers::StringView{out.str()}.contains(
      Cr::Containers::StringView{data.gfxWarning}));
  out.str("");
}

}  // namespace
CORRADE_TEST_MAIN(LoggingTest)
