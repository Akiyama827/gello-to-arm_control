// CTRL-041: the RT host's clock identity, advertised in HELLO as
//   clk=mono:<host12>:<boot12>
// so a client can tell "one CLOCK_MONOTONIC" from "same host string". Both fields are
// keyed hashes (HMAC-SHA256, first 6 bytes as hex), as machine-id(5) requires: the raw
// machine-id is never sent or logged. host = HMAC(key=machine-id, "host" label);
// boot = HMAC(key=machine-id, "boot" label + boot_id). Equal (host, boot) means one
// CLOCK_MONOTONIC epoch; anything else is a different domain. Empty result = the
// identity could not be read: the server says nothing and the client treats the domain
// as unattested (the pre-CTRL-041 host-string rule).
#pragma once
#include <openssl/hmac.h>
#include <openssl/evp.h>

#include <cstdio>
#include <string>

namespace arm_rt {

inline std::string read_id_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return {};
  char buf[80] = {};
  const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  std::string s(buf, n);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

inline std::string keyed_hex12(const std::string& key, const std::string& message) {
  unsigned char mac[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  if (!HMAC(EVP_sha256(), key.data(), int(key.size()),
            reinterpret_cast<const unsigned char*>(message.data()), message.size(), mac, &len) ||
      len < 6)
    return {};
  char hex[13];
  for (int i = 0; i < 6; ++i) std::snprintf(hex + 2 * i, 3, "%02x", mac[i]);
  return std::string(hex, 12);
}

// "clk=mono:<host12>:<boot12>", or "" when the machine-id or boot-id is unreadable.
inline std::string clock_domain_token(const std::string& machine_id_file,
                                      const std::string& boot_id_file) {
  const std::string machine = read_id_file(machine_id_file);
  const std::string boot = read_id_file(boot_id_file);
  if (machine.empty() || boot.empty()) return {};
  const std::string host = keyed_hex12(machine, "arm_rt_server:clock-domain:host:v1");
  const std::string bt = keyed_hex12(machine, "arm_rt_server:clock-domain:boot:v1:" + boot);
  if (host.empty() || bt.empty()) return {};
  return "clk=mono:" + host + ":" + bt;
}

}  // namespace arm_rt
