#ifndef ANIRA_TEST_COPY_PATH_ORACLE_GOLDEN_H
#define ANIRA_TEST_COPY_PATH_ORACLE_GOLDEN_H

// The golden transcripts of test_CopyPathOracle.cpp: what the float copy core of
// InferenceManager (process_input, process_output, hold_output, bypass_output and clear_output
// over float***, src/scheduler/InferenceManager.cpp) did at commit f7e062b, the last commit
// before that core was rewritten over anira_tensor.
//
// Produced once, on that commit's core, by running the scenarios with
//     ANIRA_COPY_ORACLE_RECORD=<file> test_scheduler --gtest_filter='CopyPathOracle.*'
// and pasting the file between the namespace braces below (test/support/copy_oracle.h,
// k_record_env). The build was quick-static (gcc, Debug, static, no engine); nothing in a
// transcript depends on the build, the machine or the timing, see "Determinism" in
// copy_oracle.h. Never record again to make a failing core pass: a difference is a change of
// what a host receives. A change that is meant is recorded for the one transcript it
// concerns, and its commit says so.
//
// Reading a transcript. The first line is what prepare left: the latencies, the ring
// capacities, the inference structs, the samples waiting per channel in every send and receive
// ring ("-" for a Static slot, which has no ring). A call is "#<n> <function> in=<counts handed
// in> out=<requests> [in-place] [null-unused] -> in=<num_input_samples after the call>
// out=<num_output_samples after the call> returned=<out when the returned pointer is
// num_output_samples> missed=<last_block_missed()> [outcome=<Core::WaitOutcome>]", then the
// rings as the call left them. Below it, one line per channel of every output the call was
// handed (io0 is the shared memory of an in-place call): the requested floats, " |", then two
// guard floats. "." is a float the call did not write. "settled:" is the rings after the gate
// opened and every submitted inference was collected; a call without it left its inferences
// at the closed gate.

#include <string_view>

namespace anira_test::oracle {

inline constexpr std::string_view k_manager_block_sizes_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_nowait in=[0] out=[0] -> in=[0] out=[0] returned=out missed=0 send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_nowait in=[8] out=[8] in-place -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_nowait in=[3] out=[3] in-place -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_nowait in=[13] out=[13] in-place -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_nowait in=[0] out=[0] in-place -> in=[0] out=[0] returned=out missed=0 send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_nowait in=[8] out=[40] -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[9/9]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_nowait in=[8] out=[40] in-place -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[17/17]
  io0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[25/25]
#15 process_nowait in=[8] out=[3] -> in=[8] out=[3] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 103 | . .
  out0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_nowait in=[20] out=[13] -> in=[20] out=[13] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 130 131 132 133 134 135 136 137 | . .
  out0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_manager_block_sizes_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_nowait in=[0] out=[0] -> in=[0] out=[0] returned=out missed=0 send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_nowait in=[8] out=[8] in-place -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_nowait in=[3] out=[3] in-place -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_nowait in=[13] out=[13] in-place -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_nowait in=[0] out=[0] in-place -> in=[0] out=[0] returned=out missed=0 send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_nowait in=[8] out=[40] -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[9/9]
  out0.c0: 48 49 50 51 52 53 54 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_nowait in=[8] out=[40] in-place -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[17/17]
  io0.c0: 48 49 50 51 52 53 54 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[25/25]
#15 process_nowait in=[8] out=[3] -> in=[8] out=[3] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 103 | . .
  out0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_nowait in=[20] out=[13] -> in=[20] out=[13] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 130 131 132 133 134 135 136 137 | . .
  out0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_manager_block_sizes_bypass = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_nowait in=[3] out=[3] -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_nowait in=[0] out=[0] -> in=[0] out=[0] returned=out missed=0 send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_nowait in=[8] out=[8] in-place -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_nowait in=[3] out=[3] in-place -> in=[3] out=[3] returned=out missed=0 send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_nowait in=[13] out=[13] in-place -> in=[13] out=[13] returned=out missed=0 send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_nowait in=[0] out=[0] in-place -> in=[0] out=[0] returned=out missed=0 send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_nowait in=[8] out=[40] -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[9/9]
  out0.c0: 71 72 73 74 75 76 77 78 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1071 1072 1073 1074 1075 1076 1077 1078 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_nowait in=[8] out=[40] in-place -> in=[8] out=[0] returned=out missed=1 send=[6/6] recv=[17/17]
  io0.c0: 79 80 81 82 83 84 85 86 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1079 1080 1081 1082 1083 1084 1085 1086 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[25/25]
#15 process_nowait in=[8] out=[3] -> in=[8] out=[3] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 103 | . .
  out0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_nowait in=[20] out=[13] -> in=[20] out=[13] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[2/2]
  out0.c0: 130 131 132 133 134 135 136 137 | . .
  out0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_manager_starved_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_nowait in=[8] out=[8] -> in=[8] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
#5 process_nowait in=[3] out=[13] -> in=[3] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
#6 process_nowait in=[8] out=[8] in-place -> in=[8] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  io0.c0: 0 0 0 0 0 0 0 0 | . .
  io0.c1: 0 0 0 0 0 0 0 0 | . .
#7 process_nowait in=[5] out=[3] -> in=[5] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 | . .
  out0.c1: 0 0 0 | . .
#8 pop_data out=[8] -> out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 49 50 51 52 53 54 55 56 | . .
  out0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data out=[20] -> out=[0] returned=out missed=1 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 70 71 72 73 74 75 76 77 | . .
  out0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_manager_starved_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_nowait in=[8] out=[8] -> in=[8] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
#5 process_nowait in=[3] out=[13] -> in=[3] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 0 0 0 0 0 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 0 0 0 0 0 | . .
#6 process_nowait in=[8] out=[8] in-place -> in=[8] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  io0.c0: 2 3 4 5 6 7 8 9 | . .
  io0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
#7 process_nowait in=[5] out=[3] -> in=[5] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
#8 pop_data out=[8] -> out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 49 50 51 52 53 54 55 56 | . .
  out0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data out=[20] -> out=[0] returned=out missed=1 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 70 71 72 73 74 75 76 77 | . .
  out0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_manager_starved_bypass = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_nowait in=[8] out=[8] -> in=[8] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 30 31 32 33 34 35 36 37 | . .
  out0.c1: 1030 1031 1032 1033 1034 1035 1036 1037 | . .
#5 process_nowait in=[3] out=[13] -> in=[3] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  out0.c0: 38 39 40 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 | . .
#6 process_nowait in=[8] out=[8] in-place -> in=[8] out=[0] returned=out missed=1 send=[0/0] recv=[2/2]
  io0.c0: 41 42 43 44 45 46 47 48 | . .
  io0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
#7 process_nowait in=[5] out=[3] -> in=[5] out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 49 50 51 | . .
  out0.c1: 1049 1050 1051 | . .
#8 pop_data out=[8] -> out=[0] returned=out missed=1 send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[5/5] recv=[0/0]
  out0.c0: 49 50 51 52 53 54 55 56 | . .
  out0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data out=[20] -> out=[0] returned=out missed=1 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 70 71 72 73 74 75 76 77 | . .
  out0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_manager_push_pop_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 push_data in=[8] -> in=[8] send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#2 push_data in=[3] -> in=[3] send=[3/3] recv=[23/23]
  settled: send=[3/3] recv=[23/23]
#3 push_data in=[5] -> in=[5] send=[0/0] recv=[23/23]
  settled: send=[0/0] recv=[31/31]
#4 pop_data out=[5] -> out=[5] returned=out missed=0 send=[0/0] recv=[26/26]
  out0.c0: 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[26/26]
#5 pop_data out=[11] -> out=[11] returned=out missed=0 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#6 pop_data out=[0] -> out=[0] returned=out missed=0 send=[0/0] recv=[15/15]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[0/0] recv=[15/15]
#7 pop_data out=[30] -> out=[0] returned=out missed=1 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#8 push_data in=[8] -> in=[8] send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#9 pop_data out=[8] -> out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[0/0]
#10 push_data in=[8] -> in=[8] send=[0/0] recv=[0/0]
#11 pop_data_until out=[8] -> out=[0] returned=out missed=1 send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#12 pop_data out=[13] -> out=[0] returned=out missed=1 send=[0/0] recv=[8/8]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_manager_push_pop_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 push_data in=[8] -> in=[8] send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#2 push_data in=[3] -> in=[3] send=[3/3] recv=[23/23]
  settled: send=[3/3] recv=[23/23]
#3 push_data in=[5] -> in=[5] send=[0/0] recv=[23/23]
  settled: send=[0/0] recv=[31/31]
#4 pop_data out=[5] -> out=[5] returned=out missed=0 send=[0/0] recv=[26/26]
  out0.c0: 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[26/26]
#5 pop_data out=[11] -> out=[11] returned=out missed=0 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#6 pop_data out=[0] -> out=[0] returned=out missed=0 send=[0/0] recv=[15/15]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[0/0] recv=[15/15]
#7 pop_data out=[30] -> out=[0] returned=out missed=1 send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#8 push_data in=[8] -> in=[8] send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#9 pop_data out=[8] -> out=[8] returned=out missed=0 send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[0/0]
#10 push_data in=[8] -> in=[8] send=[0/0] recv=[0/0]
#11 pop_data_until out=[8] -> out=[0] returned=out missed=1 send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[8/8]
#12 pop_data out=[13] -> out=[0] returned=out missed=1 send=[0/0] recv=[8/8]
  out0.c0: 17 18 19 20 21 22 23 24 0 0 0 0 0 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_manager_multi_zeros = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_nowait in=[8,0] out=[8,0] null-unused -> in=[8,0] out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_nowait in=[0,3] out=[0,3] null-unused -> in=[0,3] out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_nowait in=[8,3] out=[8,5] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_nowait in=[8,2] out=[8,2] in-place -> in=[8,2] out=[8,2] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 push_data in=[8,3] -> in=[8,3] send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#8 push_data in=[0,3] null-unused -> in=[0,3] send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#9 pop_data out=[8,3] -> out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 pop_data out=[0,3] null-unused -> out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 pop_data out=[8,0] null-unused -> out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#12 process_nowait in=[8,3] out=[40,5] -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#13 process_nowait in=[8,3] out=[40,2] in-place -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data out=[40,3] -> out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#15 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 57 58 59 60 61 62 63 64 | . .
  out0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 | . .
  out0.c2: 2057 2058 2059 2060 2061 2062 2063 2064 | . .
  out1.c0: 910130 910131 910132 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#16 reset send=[0/0/0,-] recv=[16/16/16,-]
#17 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910150 910151 910152 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#18 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910170 910171 910172 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_manager_multi_hold_last = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_nowait in=[8,0] out=[8,0] null-unused -> in=[8,0] out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_nowait in=[0,3] out=[0,3] null-unused -> in=[0,3] out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_nowait in=[8,3] out=[8,5] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_nowait in=[8,2] out=[8,2] in-place -> in=[8,2] out=[8,2] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 push_data in=[8,3] -> in=[8,3] send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#8 push_data in=[0,3] null-unused -> in=[0,3] send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#9 pop_data out=[8,3] -> out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 pop_data out=[0,3] null-unused -> out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 pop_data out=[8,0] null-unused -> out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#12 process_nowait in=[8,3] out=[40,5] -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910070 910071 910072 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#13 process_nowait in=[8,3] out=[40,2] in-place -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910120 910121 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data out=[40,3] -> out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910130 910131 910132 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#15 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 57 58 59 60 61 62 63 64 | . .
  out0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 | . .
  out0.c2: 2057 2058 2059 2060 2061 2062 2063 2064 | . .
  out1.c0: 910130 910131 910132 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#16 reset send=[0/0/0,-] recv=[16/16/16,-]
#17 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910150 910151 910152 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#18 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910170 910171 910172 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_manager_multi_bypass = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_nowait in=[8,0] out=[8,0] null-unused -> in=[8,0] out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_nowait in=[0,3] out=[0,3] null-unused -> in=[0,3] out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_nowait in=[8,3] out=[8,5] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_nowait in=[8,2] out=[8,2] in-place -> in=[8,2] out=[8,2] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 push_data in=[8,3] -> in=[8,3] send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#8 push_data in=[0,3] null-unused -> in=[0,3] send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#9 pop_data out=[8,3] -> out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 pop_data out=[0,3] null-unused -> out=[0,3] returned=out missed=0 send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910070 910071 910072 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 pop_data out=[8,0] null-unused -> out=[8,0] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#12 process_nowait in=[8,3] out=[40,5] -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 49 50 51 52 53 54 55 56 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2049 2050 2051 2052 2053 2054 2055 2056 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#13 process_nowait in=[8,3] out=[40,2] in-place -> in=[8,3] out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 57 58 59 60 61 62 63 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 2057 2058 2059 2060 2061 2062 2063 2064 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data out=[40,3] -> out=[0,0] returned=out missed=1 send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#15 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 57 58 59 60 61 62 63 64 | . .
  out0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 | . .
  out0.c2: 2057 2058 2059 2060 2061 2062 2063 2064 | . .
  out1.c0: 910130 910131 910132 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#16 reset send=[0/0/0,-] recv=[16/16/16,-]
#17 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910150 910151 910152 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#18 process_nowait in=[8,3] out=[8,3] -> in=[8,3] out=[8,3] returned=out missed=0 send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910170 910171 910172 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_manager_mono_to_stereo_bypass = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0] recv=[15/15]
#1 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0] recv=[15/15]
#2 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0] recv=[15/15]
#3 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_nowait in=[8] out=[8] -> in=[8] out=[0] returned=out missed=1 send=[5] recv=[2/2]
  out0.c0: 30 31 32 33 34 35 36 37 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
#5 process_nowait in=[3] out=[13] -> in=[3] out=[0] returned=out missed=1 send=[0] recv=[2/2]
  out0.c0: 38 39 40 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
#6 process_nowait in=[8] out=[3] -> in=[8] out=[0] returned=out missed=1 send=[0] recv=[2/2]
  out0.c0: 41 42 43 | . .
  out0.c1: 0 0 0 | . .
  settled: send=[0] recv=[34/34]
#7 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0] recv=[2/2]
  out0.c0: 39 40 41 42 43 44 45 46 | . .
  out0.c1: 1039 1040 1041 1042 1043 1044 1045 1046 | . .
  settled: send=[0] recv=[10/10]
#8 process_nowait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0] recv=[2/2]
  out0.c0: 47 48 49 50 51 52 53 54 | . .
  out0.c1: 1047 1048 1049 1050 1051 1052 1053 1054 | . .
  settled: send=[0] recv=[10/10]
)oracle";

inline constexpr std::string_view k_manager_generator = R"oracle(
latency=[15] send_capacity=[0] recv_capacity=[47] structs=4 send=[-] recv=[15]
#1 push_data in=[4] -> in=[4] send=[-] recv=[15]
  settled: send=[-] recv=[15]
#2 pop_data out=[8] -> out=[8] returned=out missed=0 send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[15]
#3 pop_data out=[8] -> out=[8] returned=out missed=0 send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 900010 | . .
  settled: send=[-] recv=[15]
#4 pop_data out=[3] -> out=[3] returned=out missed=0 send=[-] recv=[12]
  out0.c0: 900011 900012 900013 | . .
  settled: send=[-] recv=[12]
#5 process_nowait in=[4] out=[13] -> in=[4] out=[0] returned=out missed=1 send=[-] recv=[12]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[28]
#6 process_nowait in=[2] out=[8] -> in=[2] out=[8] returned=out missed=0 send=[-] recv=[7]
  out0.c0: 900051 900052 900053 900054 900055 900056 900057 900050 | . .
  settled: send=[-] recv=[15]
#7 pop_data out=[20] -> out=[0] returned=out missed=1 send=[-] recv=[15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[31]
#8 pop_data out=[8] -> out=[8] returned=out missed=0 send=[-] recv=[3]
  out0.c0: 900065 900066 900067 900060 900061 900062 900063 900064 | . .
  settled: send=[-] recv=[11]
#9 reset send=[-] recv=[15]
#10 pop_data out=[8] -> out=[8] returned=out missed=0 send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[15]
#11 pop_data out=[8] -> out=[8] returned=out missed=0 send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 900060 | . .
  settled: send=[-] recv=[15]
)oracle";

inline constexpr std::string_view k_manager_waiting_stems = R"oracle(
latency=[8] send_capacity=[15] recv_capacity=[24] structs=2 send=[0/0] recv=[8/8]
#1 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[8/8]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#2 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[0/0] recv=[8/8]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  settled: send=[0/0] recv=[8/8]
#3 process_wait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 outcome=Done send=[0/0] recv=[8/8]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  settled: send=[0/0] recv=[8/8]
#4 process_wait in=[2] out=[2] in-place -> in=[2] out=[2] returned=out missed=0 outcome=Done send=[2/2] recv=[6/6]
  io0.c0: 17 18 | . .
  io0.c1: 1017 1018 | . .
  settled: send=[2/2] recv=[6/6]
#5 process_nowait in=[13] out=[13] -> in=[13] out=[13] returned=out missed=0 send=[7/7] recv=[1/1]
  out0.c0: 19 20 21 22 23 24 25 26 27 28 29 30 31 | . .
  out0.c1: 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[7/7] recv=[1/1]
#6 push_data in=[8] -> in=[8] send=[7/7] recv=[1/1]
  settled: send=[7/7] recv=[9/9]
#7 pop_data_until out=[8] -> out=[8] returned=out missed=0 send=[7/7] recv=[1/1]
  out0.c0: 32 33 34 35 36 37 38 39 | . .
  out0.c1: 1032 1033 1034 1035 1036 1037 1038 1039 | . .
  settled: send=[7/7] recv=[1/1]
#8 push_data in=[8] -> in=[8] send=[7/7] recv=[1/1]
  settled: send=[7/7] recv=[9/9]
#9 pop_data_wait out=[8] -> out=[8] returned=out missed=0 outcome=Done send=[7/7] recv=[1/1]
  out0.c0: 40 41 42 43 44 45 46 47 | . .
  out0.c1: 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[7/7] recv=[1/1]
#10 pop_data out=[3] -> out=[0] returned=out missed=1 send=[7/7] recv=[1/1]
  out0.c0: 0 0 0 | . .
  out0.c1: 0 0 0 | . .
  settled: send=[7/7] recv=[1/1]
#11 process_wait in=[8] out=[40] -> in=[8] out=[0] returned=out missed=1 outcome=Done send=[7/7] recv=[9/9]
  out0.c0: 56 57 58 59 60 61 62 63 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1056 1057 1058 1059 1060 1061 1062 1063 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[7/7] recv=[9/9]
#12 process in=[8] out=[40] in-place -> in=[8] out=[0] returned=out missed=1 send=[7/7] recv=[17/17]
  io0.c0: 64 65 66 67 68 69 70 71 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1064 1065 1066 1067 1068 1069 1070 1071 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[7/7] recv=[17/17]
#13 pop_data_wait out=[40] -> out=[0] returned=out missed=1 outcome=Done send=[7/7] recv=[17/17]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[7/7] recv=[17/17]
#14 pop_data_until out=[40] -> out=[0] returned=out missed=1 send=[7/7] recv=[17/17]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[7/7] recv=[17/17]
#15 process_wait in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 outcome=Done send=[7/7] recv=[0/0]
  out0.c0: 57 58 59 60 61 62 63 64 | . .
  out0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 | . .
  settled: send=[7/7] recv=[8/8]
#16 process in=[8] out=[8] -> in=[8] out=[8] returned=out missed=0 send=[7/7] recv=[0/0]
  out0.c0: 73 74 75 76 77 78 79 80 | . .
  out0.c1: 1073 1074 1075 1076 1077 1078 1079 1080 | . .
  settled: send=[7/7] recv=[0/0]
)oracle";

}  // namespace anira_test::oracle

#endif  // ANIRA_TEST_COPY_PATH_ORACLE_GOLDEN_H
