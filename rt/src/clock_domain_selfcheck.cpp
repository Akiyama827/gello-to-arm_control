// CTRL-041 clock-domain token: format, determinism, separation, no raw ids, missing files.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "clock_domain.hpp"

using arm_rt::clock_domain_token;

static std::string write(const char* name, const char* body) {
  const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                           "/clock_domain_selfcheck_" + name;
  std::FILE* f = std::fopen(path.c_str(), "w");
  assert(f);
  std::fputs(body, f);
  std::fclose(f);
  return path;
}

int main() {
  const std::string m1 = write("m1", "0123456789abcdef0123456789abcdef\n");
  const std::string m2 = write("m2", "fedcba9876543210fedcba9876543210\n");
  const std::string b1 = write("b1", "11111111-2222-3333-4444-555555555555\n");
  const std::string b2 = write("b2", "66666666-7777-8888-9999-000000000000\n");

  const std::string t = clock_domain_token(m1, b1);
  assert(t.size() == 9 + 12 + 1 + 12 && t.rfind("clk=mono:", 0) == 0 && t[21] == ':');
  assert(t == clock_domain_token(m1, b1));  // deterministic
  const std::string host = t.substr(9, 12), boot = t.substr(22, 12);

  const std::string reboot = clock_domain_token(m1, b2);  // same host, new boot
  assert(reboot.substr(9, 12) == host && reboot.substr(22, 12) != boot);
  const std::string other = clock_domain_token(m2, b1);  // other host
  assert(other.substr(9, 12) != host);

  // The raw ids never appear in the token.
  assert(t.find("0123456789abcdef") == std::string::npos);
  assert(t.find("11111111") == std::string::npos);

  // Unreadable identity: nothing is claimed.
  assert(clock_domain_token("/nonexistent/machine-id", b1).empty());
  assert(clock_domain_token(m1, "/nonexistent/boot-id").empty());
  assert(clock_domain_token(write("empty", ""), b1).empty());
  std::puts("clock_domain_selfcheck: PASS");
  return 0;
}
