package org.example
import org.apache.spark.sql._

/**
 * TPC-H Query 05
 */
class TPCH_Q05 extends TPCH_Queries {

  override def TPCH_execute(sc: SparkSession, schemaProvider: TpchSchemaProvider): DataFrame = {
    // this is used to implicitly convert an RDD to a DataFrame.
    //    import sc.implicits._

//    sc.sql("select n_name,sum(l_extendedprice * (1 - l_discount)) as revenue from customer,order,lineitem,supplier,nation,region where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AFRICA' and o_orderdate >= date '1993-01-01' and o_orderdate < date '1993-01-01' + interval '1' year group by n_name order by revenue desc");
//    sc.sql("select n_name,sum(l_extendedprice * (1 - l_discount)) as revenue " +

    sc.sql("select n_name,sum(l_extendedprice * (100 - l_discount)) as revenue " +
      "from customer,order,lineitem,supplier,nation,region " +
      "where c_custkey = o_custkey " +
      "and l_orderkey = o_orderkey " +
      "and l_suppkey = s_suppkey " +
      "and c_nationkey = s_nationkey " +
      "and s_nationkey = n_nationkey " +
      "and n_regionkey = r_regionkey " +
      "and r_name = 'AFRICA' " +
      "and o_orderdate >= 19930101 " +
      "and o_orderdate < 19940101 " +
      "group by n_name order by revenue desc")

//    // Working
//    sc.sql("select l_extendedprice " +
//      "from order,lineitem " +
//      "where l_orderkey = o_orderkey " +
//      "and o_orderdate >= 19930101 " +
//      "and o_orderdate < 19940101 ")

//    // Working
//    sc.sql("select o_orderdate " +
//      "from customer, order " +
//      "where o_orderdate >= 19930101 " +
//      "and o_orderdate < 19940101 " +
//      "and c_custkey = o_custkey")

//    //working
//    sc.sql("select l_extendedprice " +
//      "from customer,order,lineitem " +
//      "where c_custkey = o_custkey " +
//      "and l_orderkey = o_orderkey " +
//      "and o_orderdate >= 19930101 " +
//      "and o_orderdate < 19940101 ")

//    // Golden test query
//    sc.sql("select l_extendedprice " +
//      "from order,lineitem " +
//      "where l_orderkey = o_orderkey " +
//        "and o_orderdate >= 19930101 " +
//      "order by l_extendedprice desc")

    /*
    val decrease = udf { (x: Double, y: Double) => x * (1 - y) }
    val increase = udf { (x: Double, y: Double) => x * (1 + y) }

    schemaProvider.lineitem.filter($"l_shipdate" <= "1998-09-02")
      .groupBy($"l_returnflag", $"l_linestatus")
      .agg(sum($"l_quantity"), sum($"l_extendedprice"),
        sum(decrease($"l_extendedprice", $"l_discount")),
        sum(increase(decrease($"l_extendedprice", $"l_discount"), $"l_tax")),
        avg($"l_quantity"), avg($"l_extendedprice"), avg($"l_discount"), count($"l_quantity"))
      .sort($"l_returnflag", $"l_linestatus")
     */
  }
}

//tbl_Filter_TD_7500_output #Row: 226645
//tbl_JOIN_INNER_TD_6810_output #Row: 226645
//tbl_JOIN_INNER_TD_5834_output #Row: 908238
//tbl_JOIN_INNER_TD_4718_output #Row: 36654
//tbl_Filter_TD_3450_output #Row: 1
//tbl_JOIN_INNER_TD_3317_output #Row: 36654
//tbl_JOIN_INNER_TD_227_output #Row: 7228
//tbl_Aggregate_TD_1400_output #Row: 5
//MOZAMBIQUE 565992005922
//ALGERIA 537123464652
//ETHIOPIA 523538844356
//MOROCCO 499581464192
//KENYA 483302225607
//tbl_Sort_TD_0244_output #Row: 5
//
//Filter_7: 7.023 ms tbl_SerializeFromObject_TD_83_input: 1500000
//JOIN_INNER_6: 41.765 ms tbl_SerializeFromObject_TD_7237_input: 150000 tbl_Filter_TD_7500_output: 226645
//JOIN_INNER_5: 321.67 ms tbl_JOIN_INNER_TD_6810_output: 226645 tbl_SerializeFromObject_TD_6780_input: 6001215
//JOIN_INNER_4: 55.108 ms tbl_JOIN_INNER_TD_5834_output: 908238 tbl_SerializeFromObject_TD_5275_input: 10000
//Filter_3: 0.004 ms tbl_SerializeFromObject_TD_4937_input: 5
//JOIN_INNER_3: 2.073 ms tbl_JOIN_INNER_TD_4718_output: 36654 tbl_SerializeFromObject_TD_4224_input: 25
//JOIN_INNER_2: 1.195 ms tbl_JOIN_INNER_TD_3317_output: 36654 tbl_Filter_TD_3450_output: 1
//Aggregate_1: 0.603 ms tbl_JOIN_INNER_TD_227_output: 7228
//Sort_0: 0.019 ms tbl_Aggregate_TD_1400_output: 5
//
//Total execution time: 429 ms
//Spark elapsed time: 6545.93ms