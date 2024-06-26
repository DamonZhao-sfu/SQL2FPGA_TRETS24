#include "ap_int.h" 
#include "xf_database/dynamic_alu_host.hpp"
#include "xf_database/enums.hpp"
#include <fstream> 
 
static void gen_pass_fcfg(uint32_t cfg[]) { 
    using namespace xf::database; 
    int n = 0; 
 
    // cond_1 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC); 
    // cond_2 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC); 
    // cond_3 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC); 
    // cond_4 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC); 
 
    uint32_t r = 0; 
    int sh = 0; 
    // cond_1 -- cond_2 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
    // cond_1 -- cond_3 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
    // cond_1 -- cond_4 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
 
    // cond_2 -- cond_3 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
    // cond_2 -- cond_4 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
 
    // cond_3 -- cond_4 
    r |= ((uint32_t)(FOP_DC << sh)); 
    sh += FilterOpWidth; 
 
    cfg[n++] = r; 
 
    // 4 true and 6 true 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)0UL; 
    cfg[n++] = (uint32_t)(1UL << 31); 
} 

void get_cfg_dat_JOIN_INNER_TD_3142_gqe_join_build(xf::database::gqe::KernelCommand& krn_cmd) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: JOIN_INNER
    // Operation: ListBuffer((c_custkey#22 = o_custkey#208))
    // Left Table: ListBuffer(c_custkey#22)
    // Right Table: ListBuffer(o_orderkey#207, o_custkey#208, o_orderdate#211, o_shippriority#214)
    // Output Table: ListBuffer(o_orderkey#207, o_orderdate#211, o_shippriority#214)
    // Node Depth: 3
    krn_cmd.setJoinOn(1);
    krn_cmd.setDualKeyOn(0);
    krn_cmd.setJoinType(0);
    krn_cmd.setJoinBuildProbe(0);
    krn_cmd.setRowIDValidEnable(0, 1, 0);
    std::vector<int8_t> enable_A;
    enable_A.push_back(0);
    enable_A.push_back(-1);
    enable_A.push_back(-1);
    krn_cmd.setScanColEnable(0, enable_A);

    std::vector<int8_t> enable_B;
    enable_B.push_back(0);
    enable_B.push_back(-1);
    enable_B.push_back(-1);
    krn_cmd.setScanColEnable(1, enable_B);
    std::vector<int8_t> enable_C;
    enable_C.push_back(0);
    enable_C.push_back(1);
    enable_C.push_back(2);
    enable_C.push_back(3);
    krn_cmd.setWriteColEnable(0, 1, enable_C);
    krn_cmd.setWriteColEnable(0, 0, enable_C);
}

void get_cfg_dat_JOIN_INNER_TD_3142_gqe_join_probe(xf::database::gqe::KernelCommand& krn_cmd) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: JOIN_INNER
    // Operation: ListBuffer((c_custkey#22 = o_custkey#208))
    // Left Table: ListBuffer(c_custkey#22)
    // Right Table: ListBuffer(o_orderkey#207, o_custkey#208, o_orderdate#211, o_shippriority#214)
    // Output Table: ListBuffer(o_orderkey#207, o_orderdate#211, o_shippriority#214)
    // Node Depth: 3
    krn_cmd.setJoinOn(1);
    krn_cmd.setDualKeyOn(0);
    krn_cmd.setJoinType(0);
    krn_cmd.setJoinBuildProbe(1);
    krn_cmd.setRowIDValidEnable(1, 1, 0);
    std::vector<int8_t> enable_A;
    enable_A.push_back(0);
    enable_A.push_back(-1);
    enable_A.push_back(-1);
    krn_cmd.setScanColEnable(0, enable_A);

    std::vector<int8_t> enable_B;
    enable_B.push_back(0);
    enable_B.push_back(-1);
    enable_B.push_back(-1);
    krn_cmd.setScanColEnable(1, enable_B);
    std::vector<int8_t> enable_C;
    enable_C.push_back(0);
    enable_C.push_back(1);
    enable_C.push_back(2);
    enable_C.push_back(3);
    krn_cmd.setWriteColEnable(0, 1, enable_C);
    krn_cmd.setWriteColEnable(0, 0, enable_C);
}

void get_meta_info_JOIN_INNER_TD_3142_gqe_join_build(MetaTable* meta, int nrow, int ncol) {
    meta->setSecID(0);
    meta->setColNum(ncol);
    for(int i=0; i<ncol; i++){
         meta->setCol(i,i,nrow);
    }
}

void get_meta_info_JOIN_INNER_TD_3142_gqe_join_probe(MetaTable* meta, int nrow, int ncol) {
    meta->setSecID(0);
    meta->setColNum(ncol);
    for(int i=0; i<ncol; i++){
         meta->setCol(i,i,nrow);
    }
}

void get_cfg_dat_JOIN_INNER_TD_2119_gqe_join(ap_uint<512>* hbuf) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: JOIN_INNER
    // Operation: ListBuffer((l_orderkey#77 = o_orderkey#207))
    // Left Table: ListBuffer(o_orderkey#207, o_orderdate#211, o_shippriority#214)
    // Right Table: ListBuffer(l_orderkey#77, l_extendedprice#82, l_discount#83)
    // Output Table: ListBuffer(o_orderdate#211, o_shippriority#214, l_orderkey#77, l_extendedprice#82, l_discount#83)
    // Node Depth: 2
    ap_uint<512>* b = hbuf;
    memset(b, 0, sizeof(ap_uint<512>) * 9);
    ap_uint<512> t = 0;

    //--------------filter--------------
    // input table a
    signed char id_a[] = {0,1,2,-1,-1,-1,-1,-1};
    for (int c = 0; c < 8; ++c) {
        t.range(56 + 8 * c + 7, 56 + 8 * c) = id_a[c];
    }
    // filter tbl_a config
    uint32_t cfga[45];
    gen_pass_fcfg(cfga);
    memcpy(&b[3], cfga, sizeof(uint32_t) * 45);

    // input table b
    signed char id_b[] = {0,1,2,-1,-1,-1,-1,-1};
    for (int c = 0; c < 8; ++c) {
        t.range(120 + 8 * c + 7, 120 + 8 * c) = id_b[c];
    }
    // filter tbl_b config
    uint32_t cfgb[45];
    gen_pass_fcfg(cfgb);
    memcpy(&b[6], cfgb, sizeof(uint32_t) * 45);

    //--------------join--------------
    //stream shuffle 1a
    ap_int<64> shuffle1a_cfg;
    shuffle1a_cfg(7, 0) = 0;
    shuffle1a_cfg(15, 8) = 1;
    shuffle1a_cfg(23, 16) = 2;
    shuffle1a_cfg(31, 24) = -1;
    shuffle1a_cfg(39, 32) = -1;
    shuffle1a_cfg(47, 40) = -1;
    shuffle1a_cfg(55, 48) = -1;
    shuffle1a_cfg(63, 56) = -1;

    //stream shuffle 1b
    ap_int<64> shuffle1b_cfg;
    shuffle1b_cfg(7, 0) = 0;
    shuffle1b_cfg(15, 8) = 2;
    shuffle1b_cfg(23, 16) = 1;
    shuffle1b_cfg(31, 24) = -1;
    shuffle1b_cfg(39, 32) = -1;
    shuffle1b_cfg(47, 40) = -1;
    shuffle1b_cfg(55, 48) = -1;
    shuffle1b_cfg(63, 56) = -1;

    // join config
    t.set_bit(0, 1);    // join
    t.set_bit(2, 0);    // dual-key
    t.range(5, 3) = 0;  // hash join flag = 0 for normal, 1 for semi, 2 for anti

    //--------------eval0--------------
    //stream shuffle 2
    ap_int<64> shuffle2_cfg;
    shuffle2_cfg(7, 0) = 6; // o_orderdate#211
    shuffle2_cfg(15, 8) = 7; // o_shippriority#214
    shuffle2_cfg(23, 16) = 12; // l_orderkey#77
    shuffle2_cfg(31, 24) = 1; // l_extendedprice#82
    shuffle2_cfg(39, 32) = 0; // l_discount#83
    shuffle2_cfg(47, 40) = -1;
    shuffle2_cfg(55, 48) = -1;
    shuffle2_cfg(63, 56) = -1;

    ap_uint<289> op_eval_0 = 0; // NOP
    // eval0: NOP
    b[1] = op_eval_0;

    //--------------eval1--------------
    //stream shuffle 3
    ap_int<64> shuffle3_cfg;
    shuffle3_cfg(7, 0) = 0; // o_orderdate#211
    shuffle3_cfg(15, 8) = 1; // o_shippriority#214
    shuffle3_cfg(23, 16) = 2; // l_orderkey#77
    shuffle3_cfg(31, 24) = 3; // l_extendedprice#82
    shuffle3_cfg(39, 32) = 4; // l_discount#83
    shuffle3_cfg(47, 40) = -1;
    shuffle3_cfg(55, 48) = -1;
    shuffle3_cfg(63, 56) = -1;

    ap_uint<289> op_eval_1 = 0; // NOP
    // eval1: NOP
    b[2] = op_eval_1;

    //--------------aggregate--------------
    //stream shuffle 4
    ap_int<64> shuffle4_cfg;
    shuffle4_cfg(7, 0) = 0; // o_orderdate#211
    shuffle4_cfg(15, 8) = 1; // o_shippriority#214
    shuffle4_cfg(23, 16) = 2; // l_orderkey#77
    shuffle4_cfg(31, 24) = 3; // l_extendedprice#82
    shuffle4_cfg(39, 32) = 4; // l_discount#83
    shuffle4_cfg(47, 40) = -1;
    shuffle4_cfg(55, 48) = -1;
    shuffle4_cfg(63, 56) = -1;

    t.set_bit(1, 0); // aggr flag

    //--------------writeout--------------
    // output table col
    t.range(191, 184) = {1*1 + 2*1 + 4*1 + 8*1 + 16*1 + 32*0 + 64*0 + 128*0};
    b[0] = t;

    //stream shuffle assignment
    b[0].range(255, 192) = shuffle1a_cfg;
    b[0].range(319, 256) = shuffle1b_cfg;
    b[0].range(383, 320) = shuffle2_cfg;
    b[0].range(447, 384) = shuffle3_cfg;
    b[0].range(511, 448) = shuffle4_cfg;
}

void get_cfg_dat_Aggregate_TD_1493_gqe_aggr(ap_uint<32>* buf) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: Aggregate
    // Operation: ListBuffer(l_orderkey#77, o_orderdate#211, o_shippriority#214, sum((l_extendedprice#82 * (100 - l_discount#83))) AS revenue#686L)
    // Input Table: ListBuffer(o_orderdate#211, o_shippriority#214, l_orderkey#77, l_extendedprice#82, l_discount#83)
    // Output Table: ListBuffer(l_orderkey#77, revenue#686L, o_orderdate#211, o_shippriority#214)
    // Node Depth: 1
    ap_uint<32>* config = buf;
    memset(config, 0, sizeof(ap_uint<32>) * 83);

    // eval0
    ap_uint<32> t;
    signed char id[] = {3,4,2,0,1,-1,-1,-1};
    for (int c = 0; c < 4; ++c) {
        t.range(8 * c + 7, 8 * c) = id[c];
    }
    config[0] = t;
    for (int c = 0; c < 4; ++c) {
        t.range(8 * c + 7, 8 * c) = id[c + 4];
    }
    config[1] = t;

    ap_uint<289> op_eval_0 = 0; // (l_extendedprice#82 * (100 - l_discount#83))
    xf::database::dynamicALUOPCompiler<uint32_t, uint32_t, uint32_t, uint32_t>("strm1*(-strm2+c2)", 0, 100, 0, 0, op_eval_0);
    for (int i = 0; i < 9; i++) {
        config[i + 2] = op_eval_0(32 * (i + 1) - 1, 32 * i);
    }
    config[11] = op_eval_0[288];

    // eval0 -> eval1
    // NO evaluation 1 in aggregation expression - eval1
    ap_int<64> shuffle1_cfg;
    shuffle1_cfg(7, 0) = 0; // l_extendedprice#82
    shuffle1_cfg(15, 8) = 1; // l_discount#83
    shuffle1_cfg(23, 16) = 2; // l_orderkey#77
    shuffle1_cfg(31, 24) = 3; // o_orderdate#211
    shuffle1_cfg(39, 32) = 4; // o_shippriority#214
    shuffle1_cfg(47, 40) = 8; // revenue#686L
    shuffle1_cfg(55, 48) = -1;
    shuffle1_cfg(63, 56) = -1;
    config[67] = shuffle1_cfg(31, 0);
    config[68] = shuffle1_cfg(63, 32);

    ap_uint<289> op_eval_1 = 0; // NOP
    // eval1: NOP
    for (int i = 0; i < 9; i++) {
        config[i + 12] = op_eval_1(32 * (i + 1) - 1, 32 * i);
    }
    config[21] = op_eval_1[288];

    // eval1 -> filter
    ap_int<64> shuffle2_cfg;
    shuffle2_cfg(7, 0) = 0; // l_extendedprice#82
    shuffle2_cfg(15, 8) = 1; // l_discount#83
    shuffle2_cfg(23, 16) = 2; // l_orderkey#77
    shuffle2_cfg(31, 24) = 3; // o_orderdate#211
    shuffle2_cfg(39, 32) = 4; // o_shippriority#214
    shuffle2_cfg(47, 40) = 5; // revenue#686L
    shuffle2_cfg(55, 48) = -1;
    shuffle2_cfg(63, 56) = -1;
    config[69] = shuffle2_cfg(31, 0);
    config[70] = shuffle2_cfg(63, 32);

    uint32_t fcfg[45];
    gen_pass_fcfg(fcfg);
    memcpy(&config[22], fcfg, sizeof(uint32_t) * 45);

    // filter -> groupBy
    ap_int<64> shuffle3_cfg;
    shuffle3_cfg(7, 0) = 2; // l_orderkey#77
    shuffle3_cfg(15, 8) = 3; // o_orderdate#211
    shuffle3_cfg(23, 16) = 4; // o_shippriority#214
    shuffle3_cfg(31, 24) = -1;
    shuffle3_cfg(39, 32) = -1;
    shuffle3_cfg(47, 40) = -1;
    shuffle3_cfg(55, 48) = -1;
    shuffle3_cfg(63, 56) = -1;
    config[71] = shuffle3_cfg(31, 0);
    config[72] = shuffle3_cfg(63, 32);

    ap_int<64> shuffle4_cfg;
    shuffle4_cfg(7, 0) = 5; // revenue#686L
    shuffle4_cfg(15, 8) = -1;
    shuffle4_cfg(23, 16) = -1;
    shuffle4_cfg(31, 24) = -1;
    shuffle4_cfg(39, 32) = -1;
    shuffle4_cfg(47, 40) = -1;
    shuffle4_cfg(55, 48) = -1;
    shuffle4_cfg(63, 56) = -1;
    config[73] = shuffle4_cfg(31, 0);
    config[74] = shuffle4_cfg(63, 32);

    ap_uint<4> aggr_op[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    aggr_op[0] = xf::database::enums::AOP_SUM;
    aggr_op[1] = xf::database::enums::AOP_SUM;
    aggr_op[2] = xf::database::enums::AOP_SUM;
    aggr_op[3] = xf::database::enums::AOP_SUM;
    aggr_op[4] = xf::database::enums::AOP_SUM;
    aggr_op[5] = xf::database::enums::AOP_SUM;
    aggr_op[6] = xf::database::enums::AOP_SUM;
    aggr_op[7] = xf::database::enums::AOP_SUM;
    config[75] = (aggr_op[7], aggr_op[6], aggr_op[5], aggr_op[4], aggr_op[3], aggr_op[2], aggr_op[1], aggr_op[0]);

    config[76] = 3; //# key col
    config[77] = 1; //# pld col
    config[78] = 0; //# aggr num

    ap_uint<8> merge[5];
    merge[0] = 0b11100000;
    merge[1] = 0;
    merge[2] = 0;
    merge[3] = 0b11100000;
    merge[4] = 0;
    config[79] = (1, merge[2], merge[1], merge[0]);
    config[80] = (0, merge[4], merge[3]);

    // aggr - demux mux direct_aggr
    config[81] = 0;

    // output - demux mux direct_aggr
    config[82] = 0xffff;
}

