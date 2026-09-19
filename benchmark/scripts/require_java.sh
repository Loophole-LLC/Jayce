# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

# Sourced by run-benchmark and benchmark-java. Stops early, with install help, when there is no
# usable JDK, instead of failing later with a cryptic "javac: command not found".

JAVA_MINIMUM=17

java_help() {
  {
    printf 'The benchmark needs a Java %s or newer JDK, but %s.\n' "$JAVA_MINIMUM" "$1"
    printf 'Install one, then run this command again:\n'
    if [[ "$(uname -s)" == Darwin ]]; then
      printf '  brew install --cask temurin\n'
    else
      printf '  sudo apt install openjdk-21-jdk          (Debian, Ubuntu)\n'
      printf '  sudo dnf install java-21-openjdk-devel   (Fedora)\n'
    fi
    printf 'or download one from https://adoptium.net\n'
  } >&2
  exit 1
}

require_java() {
  local tool version major
  for tool in java javac; do
    # On macOS /usr/bin/java exists even with no JDK installed and fails when run.
    if ! command -v "$tool" >/dev/null 2>&1 || ! "$tool" -version >/dev/null 2>&1; then
      java_help "\`$tool\` was not found"
    fi
  done
  version="$(javac -version 2>&1 | awk 'NR == 1 { print $2 }')"
  major="${version%%[.-]*}"
  if [[ "$major" == 1 ]]; then  # Java 8 and older report "1.8.0"
    major="$(printf '%s' "$version" | cut -d. -f2)"
  fi
  if [[ "$major" =~ ^[0-9]+$ ]] && (( major < JAVA_MINIMUM )); then
    java_help "this computer has Java $version"
  fi
}
