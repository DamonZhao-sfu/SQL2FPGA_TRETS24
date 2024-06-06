#include <regex> 
#include <stdint.h> 

void SW_Aggregate_TD_6132_consolidate(Table *tbl_Aggregate_TD_6132_output_preprocess, Table &tbl_Aggregate_TD_6132_output, int hpTimes) {
    int nrow = 0;
for (int p_idx = 0; p_idx < hpTimes; p_idx++) {
    int nrow_p = tbl_Aggregate_TD_6132_output_preprocess[p_idx].getNumRow();
    for (int r(0); r<nrow_p; ++r) {
        int32_t _l_orderkey77 = tbl_Aggregate_TD_6132_output_preprocess[p_idx].getInt32(r, 7);
        tbl_Aggregate_TD_6132_output.setInt32(r, 0, _l_orderkey77);
        int64_t _suml_quantity811711L = tbl_Aggregate_TD_6132_output_preprocess[p_idx].combineInt64(r, 8, 0);
        tbl_Aggregate_TD_6132_output.setInt64(r, 1, _suml_quantity811711L);
    }
    nrow += nrow_p;
}
    tbl_Aggregate_TD_6132_output.setNumRow(nrow);
    std::cout << "tbl_Aggregate_TD_6132_output #Row: " << tbl_Aggregate_TD_6132_output.getNumRow() << std::endl;
}

void SW_Filter_TD_5586(Table &tbl_Aggregate_TD_6132_output, Table &tbl_Filter_TD_5586_output) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: Filter
    // Operation: ListBuffer((isnotnull(sum(l_quantity#81)#1711L) AND (sum(l_quantity#81)#1711L > 312)))
    // Input: ListBuffer(l_orderkey#77, sum(l_quantity#81)#1711L)
    // Output: ListBuffer(l_orderkey#77)
    int r = 0;
    int nrow1 = tbl_Aggregate_TD_6132_output.getNumRow();
    for (int i = 0; i < nrow1; i++) {
        int64_t _suml_quantity811711L = tbl_Aggregate_TD_6132_output.getInt64(i, 1);
        if ((1) && (_suml_quantity811711L > 312)) {
            int32_t _l_orderkey77_t = tbl_Aggregate_TD_6132_output.getInt32(i, 0);
            tbl_Filter_TD_5586_output.setInt32(r, 0, _l_orderkey77_t);
            r++;
        }
    }
    tbl_Filter_TD_5586_output.setNumRow(r);
    std::cout << "tbl_Filter_TD_5586_output #Row: " << tbl_Filter_TD_5586_output.getNumRow() << std::endl;
}

void SW_Filter_TD_4654(Table &tbl_Aggregate_TD_6132_output, Table &tbl_Filter_TD_4654_output) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: Filter
    // Operation: ListBuffer((isnotnull(sum(l_quantity#81)#1711L) AND (sum(l_quantity#81)#1711L > 312)))
    // Input: ListBuffer(l_orderkey#77, sum(l_quantity#81)#1711L)
    // Output: ListBuffer(l_orderkey#77#1788)
    int r = 0;
    int nrow1 = tbl_Aggregate_TD_6132_output.getNumRow();
    for (int i = 0; i < nrow1; i++) {
        int64_t _suml_quantity811711L = tbl_Aggregate_TD_6132_output.getInt64(i, 1);
        if ((1) && (_suml_quantity811711L > 312)) {
            int32_t _l_orderkey771788_t = tbl_Aggregate_TD_6132_output.getInt32(i, 0);
            tbl_Filter_TD_4654_output.setInt32(r, 0, _l_orderkey771788_t);
            r++;
        }
    }
    tbl_Filter_TD_4654_output.setNumRow(r);
    std::cout << "tbl_Filter_TD_4654_output #Row: " << tbl_Filter_TD_4654_output.getNumRow() << std::endl;
}

void SW_Consolidated_JOIN_LEFTSEMI_TD_4287_output(Table& in1, Table&in2, Table& tbl_JOIN_LEFTSEMI_TD_4287_output, Table& tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: JOIN_LEFTSEMI
    // Operation: ListBuffer((o_orderkey#207 = l_orderkey#77))
    // Left Table: ListBuffer(l_orderkey#77)
    // Right Table: ListBuffer(o_orderkey#207, o_custkey#208, o_totalprice#210, o_orderdate#211)
    // Output Table: ListBuffer(o_orderkey#207, o_custkey#208, o_totalprice#210, o_orderdate#211)
    // Node Depth: 4
    int num_rows = tbl_JOIN_LEFTSEMI_TD_4287_output.getNumRow();
        for (int j = 0; j < num_rows; j++) {
          int32_t rowid_1 = tbl_JOIN_LEFTSEMI_TD_4287_output.getInt32(j, 0);
          int32_t rowid_2 = tbl_JOIN_LEFTSEMI_TD_4287_output.getInt32(j, 1);
          int64_t key = tbl_JOIN_LEFTSEMI_TD_4287_output.getInt32(j, 2);
          tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.setInt32(j, 0, in2.getInt32(rowid_1 - 1, 0));
          tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.setInt32(j, 1, in2.getInt32(rowid_1 - 1, 1));
          tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.setInt32(j, 2, in2.getInt32(rowid_1 - 1, 2));
          tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.setInt32(j, 3, in2.getInt32(rowid_1 - 1, 3));
        }
        tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.setNumRow(num_rows);
    std::cout << "tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output #Row: " << tbl_JOIN_LEFTSEMI_TD_4287_consolidated_output.getNumRow() << std::endl;
}

void SW_Consolidated_JOIN_LEFTSEMI_TD_3435_output(Table& in1, Table&in2, Table& tbl_JOIN_LEFTSEMI_TD_3435_output, Table& tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: JOIN_LEFTSEMI
    // Operation: ListBuffer((l_orderkey#77 = l_orderkey#77#1788))
    // Left Table: ListBuffer(l_orderkey#77#1788)
    // Right Table: ListBuffer(l_orderkey#77, l_quantity#81)
    // Output Table: ListBuffer(l_orderkey#77, l_quantity#81)
    // Node Depth: 3
    int num_rows = tbl_JOIN_LEFTSEMI_TD_3435_output.getNumRow();
        for (int j = 0; j < num_rows; j++) {
          int32_t rowid_1 = tbl_JOIN_LEFTSEMI_TD_3435_output.getInt32(j, 0);
          int32_t rowid_2 = tbl_JOIN_LEFTSEMI_TD_3435_output.getInt32(j, 1);
          int64_t key = tbl_JOIN_LEFTSEMI_TD_3435_output.getInt32(j, 2);
          tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output.setInt32(j, 0, in2.getInt32(rowid_1 - 1, 0));
          tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output.setInt32(j, 1, in2.getInt32(rowid_1 - 1, 1));
        }
        tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output.setNumRow(num_rows);
    std::cout << "tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output #Row: " << tbl_JOIN_LEFTSEMI_TD_3435_consolidated_output.getNumRow() << std::endl;
}

void SW_Consolidated_JOIN_INNER_TD_3608_output(Table& in1, Table&in2, Table& tbl_JOIN_INNER_TD_3608_output, Table& tbl_JOIN_INNER_TD_3608_consolidated_output) {
    // StringRowIDSubstitution: true StringRowIDBackSubstitution: false
    // Supported operation: JOIN_INNER
    // Operation: ListBuffer((c_custkey#22 = o_custkey#208))
    // Left Table: ListBuffer(c_custkey#22, c_name#23)
    // Right Table: ListBuffer(o_orderkey#207, o_custkey#208, o_totalprice#210, o_orderdate#211)
    // Output Table: ListBuffer(c_custkey#22, c_name#23, o_orderkey#207, o_totalprice#210, o_orderdate#211)
    // Node Depth: 3
    int num_rows = tbl_JOIN_INNER_TD_3608_output.getNumRow();
        for (int j = 0; j < num_rows; j++) {
          int32_t rowid_1 = tbl_JOIN_INNER_TD_3608_output.getInt32(j, 0);
          int32_t rowid_2 = tbl_JOIN_INNER_TD_3608_output.getInt32(j, 1);
          int64_t key = tbl_JOIN_INNER_TD_3608_output.getInt32(j, 2);
          tbl_JOIN_INNER_TD_3608_consolidated_output.setInt32(j, 0, in1.getInt32(rowid_2 - 1, 0));
          tbl_JOIN_INNER_TD_3608_consolidated_output.setInt32(j, 1, in1.getInt32(rowid_2 - 1, 1));
          tbl_JOIN_INNER_TD_3608_consolidated_output.setInt32(j, 2, in2.getInt32(rowid_1 - 1, 0));
          tbl_JOIN_INNER_TD_3608_consolidated_output.setInt32(j, 3, in2.getInt32(rowid_1 - 1, 2));
          tbl_JOIN_INNER_TD_3608_consolidated_output.setInt32(j, 4, in2.getInt32(rowid_1 - 1, 3));
        }
        tbl_JOIN_INNER_TD_3608_consolidated_output.setNumRow(num_rows);
    std::cout << "tbl_JOIN_INNER_TD_3608_consolidated_output #Row: " << tbl_JOIN_INNER_TD_3608_consolidated_output.getNumRow() << std::endl;
}

void SW_Consolidated_JOIN_INNER_TD_2592_output(Table& in1, Table&in2, Table& tbl_JOIN_INNER_TD_2592_output, Table& tbl_JOIN_INNER_TD_2592_consolidated_output) {
    // StringRowIDSubstitution: true StringRowIDBackSubstitution: false
    // Supported operation: JOIN_INNER
    // Operation: ListBuffer((o_orderkey#207 = l_orderkey#77))
    // Left Table: ListBuffer(c_custkey#22, c_name#23, o_orderkey#207, o_totalprice#210, o_orderdate#211)
    // Right Table: ListBuffer(l_orderkey#77, l_quantity#81)
    // Output Table: ListBuffer(c_custkey#22, c_name#23, o_orderkey#207, o_totalprice#210, o_orderdate#211, l_quantity#81)
    // Node Depth: 2
    int num_rows = tbl_JOIN_INNER_TD_2592_output.getNumRow();
        for (int j = 0; j < num_rows; j++) {
          int32_t rowid_1 = tbl_JOIN_INNER_TD_2592_output.getInt32(j, 0);
          int32_t rowid_2 = tbl_JOIN_INNER_TD_2592_output.getInt32(j, 1);
          int64_t key = tbl_JOIN_INNER_TD_2592_output.getInt32(j, 2);
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 0, in1.getInt32(rowid_2 - 1, 0));
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 1, in1.getInt32(rowid_2 - 1, 1));
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 2, in1.getInt32(rowid_2 - 1, 2));
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 3, in1.getInt32(rowid_2 - 1, 3));
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 4, in1.getInt32(rowid_2 - 1, 4));
          tbl_JOIN_INNER_TD_2592_consolidated_output.setInt32(j, 5, in2.getInt32(rowid_1 - 1, 1));
        }
        tbl_JOIN_INNER_TD_2592_consolidated_output.setNumRow(num_rows);
    std::cout << "tbl_JOIN_INNER_TD_2592_consolidated_output #Row: " << tbl_JOIN_INNER_TD_2592_consolidated_output.getNumRow() << std::endl;
}

struct SW_Aggregate_TD_1710_key {
    std::string _c_name23;
    int32_t _c_custkey22;
    int32_t _o_orderkey207;
    int32_t _o_orderdate211;
    int32_t _o_totalprice210;
    bool operator==(const SW_Aggregate_TD_1710_key& other) const { return (_c_name23 == other._c_name23) && (_c_custkey22 == other._c_custkey22) && (_o_orderkey207 == other._o_orderkey207) && (_o_orderdate211 == other._o_orderdate211) && (_o_totalprice210 == other._o_totalprice210); }
};
namespace std {
template <>
struct hash<SW_Aggregate_TD_1710_key> {
    std::size_t operator() (const SW_Aggregate_TD_1710_key& k) const {
        using std::size_t;
        using std::hash;
        using std::string;
        return (hash<string>()(k._c_name23)) + (hash<int32_t>()(k._c_custkey22)) + (hash<int32_t>()(k._o_orderkey207)) + (hash<int32_t>()(k._o_orderdate211)) + (hash<int32_t>()(k._o_totalprice210));
    }
};
}
struct SW_Aggregate_TD_1710_payload {
    int64_t _suml_quantity1714L_sum_0;
};
void SW_Aggregate_TD_1710(Table *tbl_JOIN_INNER_TD_2592_consolidated_output, Table &tbl_SerializeFromObject_TD_4246_input, Table &tbl_Aggregate_TD_1710_output, int hpTimes) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: true
    // Supported operation: Aggregate
    // Operation: ListBuffer(c_name#23, c_custkey#22, o_orderkey#207, o_orderdate#211, o_totalprice#210, sum(l_quantity#81) AS sum(l_quantity)#1714L)
    // Input: ListBuffer(c_custkey#22, c_name#23, o_orderkey#207, o_totalprice#210, o_orderdate#211, l_quantity#81)
    // Output: ListBuffer(c_name#23, c_custkey#22, o_orderkey#207, o_orderdate#211, o_totalprice#210, sum(l_quantity)#1714L)
    std::unordered_map<SW_Aggregate_TD_1710_key, SW_Aggregate_TD_1710_payload> ht1;
for (int p_idx = 0; p_idx < hpTimes; p_idx++) {
    int nrow1 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getNumRow();
    for (int i = 0; i < nrow1; i++) {
        int32_t _c_custkey22 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 0);
        std::array<char, TPCH_READ_C_NAME_LEN + 1> _c_name23 = tbl_SerializeFromObject_TD_4246_input.getcharN<char, TPCH_READ_C_NAME_LEN + 1>(tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 1), 1);
        int32_t _o_orderkey207 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 2);
        int32_t _o_totalprice210 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 3);
        int32_t _o_orderdate211 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 4);
        int32_t _l_quantity81 = tbl_JOIN_INNER_TD_2592_consolidated_output[p_idx].getInt32(i, 5);
        SW_Aggregate_TD_1710_key k{std::string(_c_name23.data()), _c_custkey22, _o_orderkey207, _o_orderdate211, _o_totalprice210};
        int64_t _suml_quantity1714L_sum_0 = _l_quantity81;
        SW_Aggregate_TD_1710_payload p{_suml_quantity1714L_sum_0};
        auto it = ht1.find(k);
        if (it != ht1.end()) {
            int64_t sum_0 = (it->second)._suml_quantity1714L_sum_0 + _suml_quantity1714L_sum_0;
            p._suml_quantity1714L_sum_0 = sum_0;
            ht1[k] = p;
        } else { 
            ht1.insert(std::make_pair(k, p));
        }
    }
}
    int r = 0;
    for (auto& it : ht1) {
        std::array<char, TPCH_READ_C_NAME_LEN + 1> _c_name23{};
        memcpy(_c_name23.data(), ((it.first)._c_name23).data(), ((it.first)._c_name23).length());
        tbl_Aggregate_TD_1710_output.setcharN<char, TPCH_READ_C_NAME_LEN + 1>(r, 0, _c_name23);
        tbl_Aggregate_TD_1710_output.setInt32(r, 1, (it.first)._c_custkey22);
        tbl_Aggregate_TD_1710_output.setInt32(r, 2, (it.first)._o_orderkey207);
        tbl_Aggregate_TD_1710_output.setInt32(r, 3, (it.first)._o_orderdate211);
        tbl_Aggregate_TD_1710_output.setInt32(r, 4, (it.first)._o_totalprice210);
        int64_t _suml_quantity1714L = (it.second)._suml_quantity1714L_sum_0;
        tbl_Aggregate_TD_1710_output.setInt64(r, 5, _suml_quantity1714L);
        ++r;
    }
    tbl_Aggregate_TD_1710_output.setNumRow(r);
    std::cout << "tbl_Aggregate_TD_1710_output #Row: " << tbl_Aggregate_TD_1710_output.getNumRow() << std::endl;
}

void SW_Sort_TD_065(Table &tbl_Aggregate_TD_1710_output, Table &tbl_Sort_TD_065_output) {
    // StringRowIDSubstitution: false StringRowIDBackSubstitution: false
    // Supported operation: Sort
    // Operation: ListBuffer(o_totalprice#210 DESC NULLS LAST, o_orderdate#211 ASC NULLS FIRST)
    // Input: ListBuffer(c_name#23, c_custkey#22, o_orderkey#207, o_orderdate#211, o_totalprice#210, sum(l_quantity)#1714L)
    // Output: ListBuffer(c_name#23, c_custkey#22, o_orderkey#207, o_orderdate#211, o_totalprice#210, sum(l_quantity)#1714L)
    struct SW_Sort_TD_065Row {
        std::string _c_name23;
        int32_t _c_custkey22;
        int32_t _o_orderkey207;
        int32_t _o_orderdate211;
        int32_t _o_totalprice210;
        int64_t _suml_quantity1714L;
    }; 

    struct {
        bool operator()(const SW_Sort_TD_065Row& a, const SW_Sort_TD_065Row& b) const { return 
 (a._o_totalprice210 > b._o_totalprice210) || 
 ((a._o_totalprice210 == b._o_totalprice210) && (a._o_orderdate211 < b._o_orderdate211)); 
}
    }SW_Sort_TD_065_order; 

    int nrow1 = tbl_Aggregate_TD_1710_output.getNumRow();
    std::vector<SW_Sort_TD_065Row> rows;
    for (int i = 0; i < nrow1; i++) {
        std::array<char, TPCH_READ_C_NAME_LEN + 1> _c_name23 = tbl_Aggregate_TD_1710_output.getcharN<char, TPCH_READ_C_NAME_LEN +1>(i, 0);
        int32_t _c_custkey22 = tbl_Aggregate_TD_1710_output.getInt32(i, 1);
        int32_t _o_orderkey207 = tbl_Aggregate_TD_1710_output.getInt32(i, 2);
        int32_t _o_orderdate211 = tbl_Aggregate_TD_1710_output.getInt32(i, 3);
        int32_t _o_totalprice210 = tbl_Aggregate_TD_1710_output.getInt32(i, 4);
        int64_t _suml_quantity1714L = tbl_Aggregate_TD_1710_output.getInt64(i, 5);
        SW_Sort_TD_065Row t = {std::string(_c_name23.data()),_c_custkey22,_o_orderkey207,_o_orderdate211,_o_totalprice210,_suml_quantity1714L};
        rows.push_back(t);
    }
    std::sort(rows.begin(), rows.end(), SW_Sort_TD_065_order);
    int r = 0;
    for (auto& it : rows) {
        std::array<char, TPCH_READ_C_NAME_LEN + 1> _c_name23{};
        memcpy(_c_name23.data(), (it._c_name23).data(), (it._c_name23).length());
        tbl_Sort_TD_065_output.setcharN<char, TPCH_READ_C_NAME_LEN +1>(r, 0, _c_name23);
        tbl_Sort_TD_065_output.setInt32(r, 1, it._c_custkey22);
        tbl_Sort_TD_065_output.setInt32(r, 2, it._o_orderkey207);
        tbl_Sort_TD_065_output.setInt32(r, 3, it._o_orderdate211);
        tbl_Sort_TD_065_output.setInt32(r, 4, it._o_totalprice210);
        tbl_Sort_TD_065_output.setInt64(r, 5, it._suml_quantity1714L);
        if (r < 10) {
            std::cout << (it._c_name23).data() << " " << it._c_custkey22 << " " << it._o_orderkey207 << " " << it._o_orderdate211 << " " << it._o_totalprice210 << " " << it._suml_quantity1714L << " " << std::endl;
        }
        ++r;
    }
    tbl_Sort_TD_065_output.setNumRow(r);
    std::cout << "tbl_Sort_TD_065_output #Row: " << tbl_Sort_TD_065_output.getNumRow() << std::endl;
}

