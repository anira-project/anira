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
# v0.5.1 (tanh-lab/tanh-lib#68). Needed since v0.5.1: thl::RCU's reader slots
# (RCU::add_reader(), read_scope(Reader)), a read section that allocates nothing, locks nothing
# and registers no thread, which a handler's plan table is read through on the driving thread
# while a control thread replaces it, and the nested read scopes and the reader-node lifetime
# fix that come with them. Needed since v0.5.0
# (tanh-lab/tanh-lib#84): tanh_add_core_library in cmake/core-library.cmake and the inline ABI
# namespace (thl::inline THL_ABI_NAMESPACE), through which CMakeLists.txt builds anira_tanh_core
# in thl::anira. Needed since v0.4.0: the strided RingBuffer calls (tanh-lab/tanh-lib#50:
# push_block / pop_block / peek_past_block with a stride) that anira_ring forwards to. Needed
# since v0.3.0: the logger carriers (tanh-lab/tanh-lib#37: platform identity, per-record flags,
# the drop count on the record) and the compiled log ceiling (#38). tanh-tooling v0.3.0, the
# same tooling tag this tree pins.
set(ANIRA_TANH_LIB_COMMIT abcd440bdde99393b7f827f110bc59f4a0ed49e0)
