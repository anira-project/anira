#ifndef ANIRA_TEST_HANDLER_COPY_ORACLE_GOLDEN_H
#define ANIRA_TEST_HANDLER_COPY_ORACLE_GOLDEN_H

// The golden transcripts of test_HandlerCopyOracle.cpp: what the _f32 Hard entries
// (src/capi/handler.cpp) did over the float copy core of InferenceManager at commit f7e062b,
// the last commit before that core was rewritten over anira_tensor.
//
// Produced once, on that commit's core, by running the scenarios with
//     ANIRA_COPY_ORACLE_RECORD=<file> test_abi --gtest_filter='AbiHandlerCopyOracle.*'
// and pasting the file between the namespace braces below (test/support/copy_oracle.h,
// k_record_env). The build was quick-static (gcc, Debug, static, no engine); nothing in a
// transcript depends on the build, the machine or the timing, see "Determinism" in
// copy_oracle.h. Never record again to make a failing core pass: a difference is a change of
// what a host receives. A change that is meant is recorded for the one transcript it
// concerns, and its commit says so.
//
// Reading a transcript. The first line is what prepare left: the latencies, the ring
// capacities, the inference structs, the samples waiting per channel in every send and receive
// ring ("-" for a Static slot, which has no ring). A call is "#<n> <entry> <arguments>
// [in-place] [null-unused] -> <OK | MISSED | status(<n>)> <delivered=<count | unset | null> of
// a single form, num_out=<the caller's array after the call> of a multi form>
// rt_error=<anira_handler_rt_error>", then the rings as the call left them. Below it, one line
// per channel of every output the call was handed (io<slot> is the memory of an in-place
// call): the requested floats, " |", then two guard floats. "." is a float the call did not
// write. "settled:" is the rings after the gate opened and every submitted inference was
// collected; a call without it left its inferences at the closed gate.

#include <string_view>

namespace anira_test::oracle {

inline constexpr std::string_view k_handler_block_sizes_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_f32 slot=0 in=0 out=0 -> OK delivered=0 rt_error=OK send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_f32_inplace slot=0 n=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_f32_inplace slot=0 n=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_f32_inplace slot=0 n=0 -> OK delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process_f32 slot=0 in=8 out=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_f32 slot=0 in=8 out=40 -> MISSED delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[0/0]
  io0.c0: 65 66 67 68 69 70 71 72 | . .
  io0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  settled: send=[6/6] recv=[8/8]
#15 process_f32 slot=0 in=8 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 103 | . .
  io0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_f32 slot=0 in=20 out=13 -> OK delivered=13 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  io0.c0: 130 131 132 133 134 135 136 137 | . .
  io0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_handler_block_sizes_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_f32 slot=0 in=0 out=0 -> OK delivered=0 rt_error=OK send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_f32_inplace slot=0 n=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_f32_inplace slot=0 n=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_f32_inplace slot=0 n=0 -> OK delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process_f32 slot=0 in=8 out=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_f32 slot=0 in=8 out=40 -> MISSED delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 48 49 50 51 52 53 54 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[0/0]
  io0.c0: 65 66 67 68 69 70 71 72 | . .
  io0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  settled: send=[6/6] recv=[8/8]
#15 process_f32 slot=0 in=8 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 103 | . .
  io0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_f32 slot=0 in=20 out=13 -> OK delivered=13 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  io0.c0: 130 131 132 133 134 135 136 137 | . .
  io0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_handler_block_sizes_bypass = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[3/3] recv=[12/12]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
  settled: send=[3/3] recv=[12/12]
#4 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 5 6 7 | . .
  out0.c1: 1005 1006 1007 | . .
  settled: send=[6/6] recv=[9/9]
#5 process_f32 slot=0 in=3 out=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  out0.c0: 8 9 10 | . .
  out0.c1: 1008 1009 1010 | . .
  settled: send=[1/1] recv=[14/14]
#6 process_f32 slot=0 in=0 out=0 -> OK delivered=0 rt_error=OK send=[1/1] recv=[14/14]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[1/1] recv=[14/14]
#7 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 11 12 13 14 15 16 17 18 19 20 21 22 23 | . .
  out0.c1: 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 | . .
  settled: send=[6/6] recv=[9/9]
#8 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 24 25 26 27 28 29 30 31 | . .
  io0.c1: 1024 1025 1026 1027 1028 1029 1030 1031 | . .
  settled: send=[6/6] recv=[9/9]
#9 process_f32_inplace slot=0 n=3 -> OK delivered=3 rt_error=OK send=[1/1] recv=[6/6]
  io0.c0: 32 33 34 | . .
  io0.c1: 1032 1033 1034 | . .
  settled: send=[1/1] recv=[14/14]
#10 process_f32_inplace slot=0 n=13 -> OK delivered=13 rt_error=OK send=[6/6] recv=[1/1]
  io0.c0: 35 36 37 38 39 40 41 42 43 44 45 46 47 | . .
  io0.c1: 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 | . .
  settled: send=[6/6] recv=[9/9]
#11 process_f32_inplace slot=0 n=0 -> OK delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  io0.c0: | . .
  io0.c1: | . .
  settled: send=[6/6] recv=[9/9]
#12 process_f32 slot=0 in=8 out=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[1/1]
  out0.c0: 48 49 50 51 52 53 54 55 | . .
  out0.c1: 1048 1049 1050 1051 1052 1053 1054 1055 | . .
  settled: send=[6/6] recv=[9/9]
#13 process_f32 slot=0 in=8 out=40 -> MISSED delivered=0 rt_error=OK send=[6/6] recv=[9/9]
  out0.c0: 71 72 73 74 75 76 77 78 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1071 1072 1073 1074 1075 1076 1077 1078 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[6/6] recv=[17/17]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=null rt_error=OK send=[6/6] recv=[0/0]
  io0.c0: 65 66 67 68 69 70 71 72 | . .
  io0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  settled: send=[6/6] recv=[8/8]
#15 process_f32 slot=0 in=8 out=3 -> OK delivered=3 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 78 79 80 | . .
  out0.c1: 1078 1079 1080 | . .
  settled: send=[6/6] recv=[8/8]
#16 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[6/6] recv=[0/0]
  out0.c0: 81 82 83 84 85 86 87 88 | . .
  out0.c1: 1081 1082 1083 1084 1085 1086 1087 1088 | . .
  settled: send=[6/6] recv=[8/8]
#17 reset send=[0/0] recv=[15/15]
#18 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#19 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 103 | . .
  io0.c1: 0 0 0 0 0 0 0 1103 | . .
  settled: send=[0/0] recv=[15/15]
#20 process_f32 slot=0 in=20 out=13 -> OK delivered=13 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 104 105 106 107 108 109 110 111 112 113 114 115 116 | . .
  out0.c1: 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 | . .
  settled: send=[7/7] recv=[10/10]
#21 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  out0.c0: 117 118 124 125 126 127 128 129 | . .
  out0.c1: 1117 1118 1124 1125 1126 1127 1128 1129 | . .
  settled: send=[7/7] recv=[10/10]
#22 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[7/7] recv=[2/2]
  io0.c0: 130 131 132 133 134 135 136 137 | . .
  io0.c1: 1130 1131 1132 1133 1134 1135 1136 1137 | . .
  settled: send=[7/7] recv=[10/10]
)oracle";

inline constexpr std::string_view k_handler_starved_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 1 | . .
  io0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_f32 slot=0 in=8 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
#5 process_f32 slot=0 in=3 out=13 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
#6 process_f32_inplace slot=0 n=8 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  io0.c0: 0 0 0 0 0 0 0 0 | . .
  io0.c1: 0 0 0 0 0 0 0 0 | . .
#7 process_f32 slot=0 in=5 out=3 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 | . .
  out0.c1: 0 0 0 | . .
#8 pop_data_f32 slot=0 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  io0.c0: 49 50 51 52 53 54 55 56 | . .
  io0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data_f32 slot=0 out=20 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  io0.c0: 70 71 72 73 74 75 76 77 | . .
  io0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_handler_starved_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 1 | . .
  io0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_f32 slot=0 in=8 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
#5 process_f32 slot=0 in=3 out=13 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 0 0 0 0 0 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 0 0 0 0 0 | . .
#6 process_f32_inplace slot=0 n=8 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  io0.c0: 2 3 4 5 6 7 8 9 | . .
  io0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
#7 process_f32 slot=0 in=5 out=3 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 | . .
  out0.c1: 1002 1003 1004 | . .
#8 pop_data_f32 slot=0 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  io0.c0: 49 50 51 52 53 54 55 56 | . .
  io0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data_f32 slot=0 out=20 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  io0.c0: 70 71 72 73 74 75 76 77 | . .
  io0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_handler_starved_bypass = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#2 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[7/7]
  io0.c0: 0 0 0 0 0 0 0 1 | . .
  io0.c1: 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#3 process_f32 slot=0 in=13 out=13 -> OK delivered=13 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 2 3 4 5 6 7 8 9 10 11 12 13 14 | . .
  out0.c1: 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 | . .
#4 process_f32 slot=0 in=8 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 30 31 32 33 34 35 36 37 | . .
  out0.c1: 1030 1031 1032 1033 1034 1035 1036 1037 | . .
#5 process_f32 slot=0 in=3 out=13 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  out0.c0: 38 39 40 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 | . .
#6 process_f32_inplace slot=0 n=8 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[2/2]
  io0.c0: 41 42 43 44 45 46 47 48 | . .
  io0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
#7 process_f32 slot=0 in=5 out=3 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 49 50 51 | . .
  out0.c1: 1049 1050 1051 | . .
#8 pop_data_f32 slot=0 out=8 -> MISSED delivered=0 rt_error=OK send=[5/5] recv=[2/2]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[5/5] recv=[34/34]
#9 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  settled: send=[5/5] recv=[8/8]
#10 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[5/5] recv=[0/0]
  io0.c0: 49 50 51 52 53 54 55 56 | . .
  io0.c1: 1049 1050 1051 1052 1053 1054 1055 1056 | . .
  settled: send=[5/5] recv=[8/8]
#11 reset send=[0/0] recv=[15/15]
#12 pop_data_f32 slot=0 out=20 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#13 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#14 process_f32_inplace slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0] recv=[0/0]
  io0.c0: 70 71 72 73 74 75 76 77 | . .
  io0.c1: 1070 1071 1072 1073 1074 1075 1076 1077 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_handler_push_pop_zeros = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#2 push_data_f32 slot=0 in=3 -> OK rt_error=OK send=[3/3] recv=[23/23]
  settled: send=[3/3] recv=[23/23]
#3 push_data_f32 slot=0 in=5 -> OK rt_error=OK send=[0/0] recv=[23/23]
  settled: send=[0/0] recv=[31/31]
#4 pop_data_f32 slot=0 out=5 -> OK delivered=5 rt_error=OK send=[0/0] recv=[26/26]
  out0.c0: 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[26/26]
#5 pop_data_f32 slot=0 out=11 -> OK delivered=11 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#6 pop_data_f32 slot=0 out=0 -> OK delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[0/0] recv=[15/15]
#7 pop_data_f32 slot=0 out=30 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#8 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#9 pop_data_f32 slot=0 out=8 -> OK delivered=null rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[0/0]
#10 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[0/0]
#11 pop_data_f32 slot=0 out=8 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
#12 pop_data_f32 slot=0 out=13 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[8/8]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_handler_push_pop_hold_last = R"oracle(
latency=[15] send_capacity=[15] recv_capacity=[47] structs=4 send=[0/0] recv=[15/15]
#1 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#2 push_data_f32 slot=0 in=3 -> OK rt_error=OK send=[3/3] recv=[23/23]
  settled: send=[3/3] recv=[23/23]
#3 push_data_f32 slot=0 in=5 -> OK rt_error=OK send=[0/0] recv=[23/23]
  settled: send=[0/0] recv=[31/31]
#4 pop_data_f32 slot=0 out=5 -> OK delivered=5 rt_error=OK send=[0/0] recv=[26/26]
  out0.c0: 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[26/26]
#5 pop_data_f32 slot=0 out=11 -> OK delivered=11 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 1 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 1001 | . .
  settled: send=[0/0] recv=[15/15]
#6 pop_data_f32 slot=0 out=0 -> OK delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: | . .
  out0.c1: | . .
  settled: send=[0/0] recv=[15/15]
#7 pop_data_f32 slot=0 out=30 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[15/15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[15/15]
#8 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[15/15]
  settled: send=[0/0] recv=[23/23]
#9 pop_data_f32 slot=0 out=8 -> OK delivered=null rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[0/0]
#10 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0] recv=[0/0]
#11 pop_data_f32 slot=0 out=8 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[0/0]
  out0.c0: 17 18 19 20 21 22 23 24 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  settled: send=[0/0] recv=[8/8]
#12 pop_data_f32 slot=0 out=13 -> MISSED delivered=0 rt_error=OK send=[0/0] recv=[8/8]
  out0.c0: 17 18 19 20 21 22 23 24 0 0 0 0 0 | . .
  out0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 0 0 0 0 0 | . .
  settled: send=[0/0] recv=[8/8]
)oracle";

inline constexpr std::string_view k_handler_multi_zeros = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_f32_multi in=[8,0] out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_f32_multi in=[0,3] out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_f32_multi in=[8,3] out=[8,5] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_f32_multi in=[8,2] out=[8,2] in-place -> OK num_out=[8,2] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 process_f32 slot=1 in=3 out=3 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#8 process_f32 slot=1 in=3 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#9 process_f32_inplace slot=1 n=2 -> OK delivered=2 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io1.c0: 910060 910061 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 push_data_f32_multi in=[8,3] -> OK rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#12 push_data_f32_multi in=[0,3] null-unused -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#13 push_data_f32 slot=1 in=3 -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data_f32_multi out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#15 pop_data_f32_multi out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#16 pop_data_f32_multi out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#17 pop_data_f32 slot=1 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out1.c0: 910110 910111 910112 . . | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#18 process_f32_multi in=[8,3] out=[40,5] -> MISSED num_out=[40,5] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#19 process_f32_multi in=[8,3] out=[40,2] in-place -> MISSED num_out=[40,2] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#20 pop_data_f32_multi out=[40,3] -> MISSED num_out=[40,3] rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#21 pop_data_f32 slot=0 out=40 -> MISSED delivered=0 rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#22 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 65 66 67 68 69 70 71 72 | . .
  out0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  out0.c2: 2065 2066 2067 2068 2069 2070 2071 2072 | . .
  out1.c0: 910190 910191 910192 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#23 reset send=[0/0/0,-] recv=[16/16/16,-]
#24 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910220 910221 910222 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#25 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910240 910241 910242 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_handler_multi_hold_last = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_f32_multi in=[8,0] out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_f32_multi in=[0,3] out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_f32_multi in=[8,3] out=[8,5] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_f32_multi in=[8,2] out=[8,2] in-place -> OK num_out=[8,2] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 process_f32 slot=1 in=3 out=3 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#8 process_f32 slot=1 in=3 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#9 process_f32_inplace slot=1 n=2 -> OK delivered=2 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io1.c0: 910060 910061 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 push_data_f32_multi in=[8,3] -> OK rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#12 push_data_f32_multi in=[0,3] null-unused -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#13 push_data_f32 slot=1 in=3 -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data_f32_multi out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#15 pop_data_f32_multi out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#16 pop_data_f32_multi out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#17 pop_data_f32 slot=1 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out1.c0: 910110 910111 910112 . . | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#18 process_f32_multi in=[8,3] out=[40,5] -> MISSED num_out=[40,5] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 41 42 43 44 45 46 47 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910110 910111 910112 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#19 process_f32_multi in=[8,3] out=[40,2] in-place -> MISSED num_out=[40,2] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 41 42 43 44 45 46 47 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910180 910181 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#20 pop_data_f32_multi out=[40,3] -> MISSED num_out=[40,3] rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 41 42 43 44 45 46 47 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910190 910191 910192 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#21 pop_data_f32 slot=0 out=40 -> MISSED delivered=0 rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 41 42 43 44 45 46 47 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#22 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 65 66 67 68 69 70 71 72 | . .
  out0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  out0.c2: 2065 2066 2067 2068 2069 2070 2071 2072 | . .
  out1.c0: 910190 910191 910192 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#23 reset send=[0/0/0,-] recv=[16/16/16,-]
#24 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910220 910221 910222 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#25 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910240 910241 910242 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_handler_multi_bypass = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910010 910011 910012 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_f32_multi in=[8,0] out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 1 2 3 4 5 6 7 8 | . .
  out0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  out0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_f32_multi in=[0,3] out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910020 910021 910022 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#5 process_f32_multi in=[8,3] out=[8,5] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 9 10 11 12 13 14 15 16 | . .
  out0.c1: 1009 1010 1011 1012 1013 1014 1015 1016 | . .
  out0.c2: 2009 2010 2011 2012 2013 2014 2015 2016 | . .
  out1.c0: 910020 910021 910022 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#6 process_f32_multi in=[8,2] out=[8,2] in-place -> OK num_out=[8,2] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  io0.c0: 17 18 19 20 21 22 23 24 | . .
  io0.c1: 1017 1018 1019 1020 1021 1022 1023 1024 | . .
  io0.c2: 2017 2018 2019 2020 2021 2022 2023 2024 | . .
  out1.c0: 910050 910051 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 process_f32 slot=1 in=3 out=3 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#8 process_f32 slot=1 in=3 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910060 910061 910052 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#9 process_f32_inplace slot=1 n=2 -> OK delivered=2 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io1.c0: 910060 910061 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#10 process_f32 slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 push_data_f32_multi in=[8,3] -> OK rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#12 push_data_f32_multi in=[0,3] null-unused -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#13 push_data_f32 slot=1 in=3 -> OK rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#14 pop_data_f32_multi out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#15 pop_data_f32_multi out=[0,3] null-unused -> OK num_out=[0,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out1.c0: 910110 910111 910112 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#16 pop_data_f32_multi out=[8,0] null-unused -> OK num_out=[8,0] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 41 42 43 44 45 46 47 48 | . .
  out0.c1: 1041 1042 1043 1044 1045 1046 1047 1048 | . .
  out0.c2: 2041 2042 2043 2044 2045 2046 2047 2048 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#17 pop_data_f32 slot=1 out=5 -> OK delivered=3 rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out1.c0: 910110 910111 910112 . . | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#18 process_f32_multi in=[8,3] out=[40,5] -> MISSED num_out=[40,5] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 57 58 59 60 61 62 63 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1057 1058 1059 1060 1061 1062 1063 1064 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2057 2058 2059 2060 2061 2062 2063 2064 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#19 process_f32_multi in=[8,3] out=[40,2] in-place -> MISSED num_out=[40,2] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 65 66 67 68 69 70 71 72 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  io0.c2: 2065 2066 2067 2068 2069 2070 2071 2072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#20 pop_data_f32_multi out=[40,3] -> MISSED num_out=[40,3] rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#21 pop_data_f32 slot=0 out=40 -> MISSED delivered=0 rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#22 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 65 66 67 68 69 70 71 72 | . .
  out0.c1: 1065 1066 1067 1068 1069 1070 1071 1072 | . .
  out0.c2: 2065 2066 2067 2068 2069 2070 2071 2072 | . .
  out1.c0: 910190 910191 910192 | . .
  settled: send=[0/0/0,-] recv=[8/8/8,-]
#23 reset send=[0/0/0,-] recv=[16/16/16,-]
#24 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910220 910221 910222 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#25 process_f32_multi in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[8/8/8,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910240 910241 910242 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
)oracle";

inline constexpr std::string_view k_handler_waiting_twins = R"oracle(
latency=[16,0] send_capacity=[15,0] recv_capacity=[48,0] structs=4 send=[0/0/0,-] recv=[16/16/16,-]
#1 process_f32_wait slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#2 process_f32_wait slot=0 in=8 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  out0.c1: 0 0 0 0 0 0 0 0 | . .
  out0.c2: 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#3 process_f32_inplace_wait slot=0 n=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 1 2 3 4 5 6 7 8 | . .
  io0.c1: 1001 1002 1003 1004 1005 1006 1007 1008 | . .
  io0.c2: 2001 2002 2003 2004 2005 2006 2007 2008 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#4 process_f32_inplace_wait slot=0 n=3 -> OK delivered=3 rt_error=OK send=[3/3/3,-] recv=[13/13/13,-]
  io0.c0: 9 10 11 | . .
  io0.c1: 1009 1010 1011 | . .
  io0.c2: 2009 2010 2011 | . .
  settled: send=[3/3/3,-] recv=[13/13/13,-]
#5 process_f32_multi_wait in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[3/3/3,-] recv=[13/13/13,-]
  out0.c0: 12 13 14 15 16 17 18 19 | . .
  out0.c1: 1012 1013 1014 1015 1016 1017 1018 1019 | . .
  out0.c2: 2012 2013 2014 2015 2016 2017 2018 2019 | . .
  out1.c0: 910050 910051 910052 | . .
  settled: send=[3/3/3,-] recv=[13/13/13,-]
#6 process_f32_multi_wait in=[5,3] out=[5,5] in-place -> OK num_out=[5,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  io0.c0: 20 21 22 23 24 | . .
  io0.c1: 1020 1021 1022 1023 1024 | . .
  io0.c2: 2020 2021 2022 2023 2024 | . .
  out1.c0: 910060 910061 910062 . . | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#7 push_data_f32 slot=0 in=8 -> OK rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#8 pop_data_f32_wait slot=0 out=8 -> OK delivered=8 rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 25 26 27 28 29 30 31 32 | . .
  out0.c1: 1025 1026 1027 1028 1029 1030 1031 1032 | . .
  out0.c2: 2025 2026 2027 2028 2029 2030 2031 2032 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#9 push_data_f32_multi in=[8,3] -> OK rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#10 pop_data_f32_multi_wait out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[16/16/16,-]
  out0.c0: 33 34 35 36 37 38 39 40 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 | . .
  out1.c0: 910090 910091 910092 | . .
  settled: send=[0/0/0,-] recv=[16/16/16,-]
#11 process_f32_wait slot=0 in=8 out=40 -> MISSED delivered=0 rt_error=OK send=[0/0/0,-] recv=[24/24/24,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[24/24/24,-]
#12 process_f32_multi_wait in=[8,3] out=[40,5] -> MISSED num_out=[40,5] rt_error=OK send=[0/0/0,-] recv=[32/32/32,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910120 910121 910122 . . | . .
  settled: send=[0/0/0,-] recv=[32/32/32,-]
#13 pop_data_f32_wait slot=0 out=40 -> MISSED delivered=0 rt_error=OK send=[0/0/0,-] recv=[32/32/32,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[0/0/0,-] recv=[32/32/32,-]
#14 pop_data_f32_multi_wait out=[40,3] -> MISSED num_out=[40,3] rt_error=OK send=[0/0/0,-] recv=[32/32/32,-]
  out0.c0: 33 34 35 36 37 38 39 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c1: 1033 1034 1035 1036 1037 1038 1039 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out0.c2: 2033 2034 2035 2036 2037 2038 2039 2040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  out1.c0: 910120 910121 910122 | . .
  settled: send=[0/0/0,-] recv=[32/32/32,-]
#15 process_f32_multi_wait in=[8,3] out=[8,3] -> OK num_out=[8,3] rt_error=OK send=[0/0/0,-] recv=[0/0/0,-]
  out0.c0: 73 74 75 76 77 78 79 80 | . .
  out0.c1: 1073 1074 1075 1076 1077 1078 1079 1080 | . .
  out0.c2: 2073 2074 2075 2076 2077 2078 2079 2080 | . .
  out1.c0: 910150 910151 910152 | . .
  settled: send=[0/0/0,-] recv=[0/0/0,-]
)oracle";

inline constexpr std::string_view k_handler_generator = R"oracle(
latency=[15] send_capacity=[0] recv_capacity=[47] structs=4 send=[-] recv=[15]
#1 push_data_f32 slot=0 in=4 -> OK rt_error=OK send=[-] recv=[15]
  settled: send=[-] recv=[15]
#2 pop_data_f32 slot=0 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[15]
#3 pop_data_f32 slot=0 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 900010 | . .
  settled: send=[-] recv=[15]
#4 pop_data_f32 slot=0 out=3 -> OK delivered=3 rt_error=OK send=[-] recv=[12]
  out0.c0: 900011 900012 900013 | . .
  settled: send=[-] recv=[12]
#5 process_f32 slot=0 in=4 out=13 -> MISSED delivered=0 rt_error=OK send=[-] recv=[12]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[28]
#6 process_f32 slot=0 in=2 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[7]
  out0.c0: 900051 900052 900053 900054 900055 900056 900057 900050 | . .
  settled: send=[-] recv=[15]
#7 pop_data_f32 slot=0 out=20 -> MISSED delivered=0 rt_error=OK send=[-] recv=[15]
  out0.c0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[31]
#8 pop_data_f32 slot=0 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[3]
  out0.c0: 900065 900066 900067 900060 900061 900062 900063 900064 | . .
  settled: send=[-] recv=[11]
#9 reset send=[-] recv=[15]
#10 pop_data_f32 slot=0 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 0 | . .
  settled: send=[-] recv=[15]
#11 pop_data_f32 slot=0 out=8 -> OK delivered=8 rt_error=OK send=[-] recv=[7]
  out0.c0: 0 0 0 0 0 0 0 900060 | . .
  settled: send=[-] recv=[15]
)oracle";

}  // namespace anira_test::oracle

#endif  // ANIRA_TEST_HANDLER_COPY_ORACLE_GOLDEN_H
