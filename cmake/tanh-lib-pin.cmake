# ==============================================================================
# anira · cmake/tanh-lib-pin.cmake — the tanh-lib commit, in one place.
#
#   ANIRA_TANH_LIB_REPOSITORY   the repository anira fetches tanh-lib from
#   ANIRA_TANH_LIB_COMMIT       the commit it pins
#
# Read by CMakeLists.txt (the private copy of tanh-lib's core that libanira absorbs) and by
# test/contracts/install/CMakeLists.txt (the host's own tanh-lib of consumer_tanh_first), so the
# two can never pin different commits. anira and the pinned tanh-lib must carry the same
# tanh-tooling tag in cmake/tanh/ (modules-version.cmake warns otherwise).
# ==============================================================================
include_guard(GLOBAL)

set(ANIRA_TANH_LIB_REPOSITORY https://github.com/tanh-lab/tanh-lib.git)
# v0.4.0 plus tanh-tooling v0.3.0 (tanh-lab/tanh-lib#73; nothing in the core changed since
# v0.4.0). Needed since v0.4.0: the strided RingBuffer calls (tanh-lab/tanh-lib#50: push_block /
# pop_block / peek_past_block with a stride) that anira_ring forwards to. Needed since v0.3.0:
# the logger carriers (tanh-lab/tanh-lib#37: platform identity, per-record flags, the drop count
# on the record) and the TANH_LOG_COMPILED_MAX_LEVEL option (#38). tanh-tooling v0.3.0, the same
# tooling tag this tree pins.
set(ANIRA_TANH_LIB_COMMIT 7ae7c690d57806acc56dd4b5ffbba50551913966)
