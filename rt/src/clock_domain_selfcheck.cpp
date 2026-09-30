// CTRL-041 clock-domain token: format, determinism, separation, no raw ids, missing files.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "clock_domain.hpp"

using arm_rt::clock_domain_token;

static std::vector<std::string> created;

static std::string write(const char* name, const char* body) {
  std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                     "/clock_domain_selfcheck_" + name + "_XXXXXX";
  const int fd = mkstemp(&path[0]);  // unique per run: concurrent ctest runs cannot race
  assert(fd >= 0);
  const ssize_t n = ::write(fd, body, std::strlen(body));
  assert(n == ssize_t(std::strlen(body)));
  close(fd);
  created.push_back(path);
  return path;
}

int main() {
  const std::string m1 = write("m1", "0123456789abcdef0123456789abcdef\n");
  const std::string m2 = write("m2", "fedcba9876543210fedcba9876543210\n");
  const std::string b1 = write("b1", "11111111-2222-3333-4444-555555555555\n");
  const std::string b2 = write("b2", "66666666-7777-8888-9999-000000000000\n");

  const std::string t = clock_domain_token(m1, b1);
  assert(t.size() == 4 + 12 + 1 + 12 && t.rfind("clk=", 0) == 0 && t[16] == ':');
  assert(t == clock_domain_token(m1, b1));  // deterministic
  const std::string host = t.substr(4, 12), boot = t.substr(17, 12);

  const std::string reboot = clock_domain_token(m1, b2);  // same host, new boot
  assert(reboot.substr(4, 12) == host && reboot.substr(17, 12) != boot);
  const std::string other = clock_domain_token(m2, b1);  // other host
  assert(other.substr(4, 12) != host);

  // The raw ids never appear in the token.
  assert(t.find("0123456789abcdef") == std::string::npos);
  assert(t.find("11111111") == std::string::npos);

  // Unreadable identity: nothing is claimed.
  assert(clock_domain_token("/nonexistent/machine-id", b1).empty());
  assert(clock_domain_token(m1, "/nonexistent/boot-id").empty());
  assert(clock_domain_token(write("empty", ""), b1).empty());
  assert(clock_domain_token(write("placeholder", "uninitialized\n"), b1).empty());
  assert(clock_domain_token(write("upper", "0123456789ABCDEF0123456789ABCDEF\n"), b1).empty());
  for (const auto& p : created) std::remove(p.c_str());
  std::puts("clock_domain_selfcheck: PASS");
  return 0;
}
