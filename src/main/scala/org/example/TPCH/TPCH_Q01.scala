package org.example
import org.apache.spark.sql._


/**
 * TPC-H Query 01
 */
class TPCH_Q01 extends TPCH_Queries {

  override def TPCH_execute(sc: SparkSession, schemaProvider: TpchSchemaProvider): DataFrame = {
//    sc.sql("select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order " +
//      "from lineitem " +
//      "where l_shipdate <= date '1998-12-01' - interval '120' day " +
//      "group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;")

//    sc.sql("cache table customer;")
//    sc.sql("cache table lineitem;")
//    sc.sql("cache table nation;")
//    sc.sql("cache table region;")
//    sc.sql("cache table order;")
//    sc.sql("cache table part;")
//    sc.sql("cache table partsupp;")
//    sc.sql("cache table supplier;")

    sc.sql("select l_returnflag, l_linestatus, " +
        "sum(l_quantity) as sum_qty, " +
        "sum(l_extendedprice) as sum_base_price, " +
        "sum(l_extendedprice * (100 - l_discount)) as sum_disc_price, " +
        "sum(l_extendedprice * (100 - l_discount) * (100 + l_tax)) as sum_charge, " +
        "avg(l_quantity) as avg_qty, " +
        "avg(l_extendedprice) as avg_price, " +
        "avg(l_discount) as avg_disc, " +
        "count(*) as count_order " +
      "from lineitem " +
      "where l_shipdate <= 19980803 " +
      "group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;")

//    sc.sql("select l_returnflag, " +
//        "sum(l_quantity) as sum_qty " +
//      "from lineitem " +
//      "where l_shipdate <= 19980803 " +
//      "group by l_returnflag " +
//      "order by l_returnflag;" )
  }
}

//65 37734107
//78 73790110
//82 37719753
//
//65 37734107
//78 73790110
//82 37719753

//tbl_Filter_TD_2351_output #Row: 1272911
//  tbl_JOIN_INNER_TD_1538_output #Row: 5093221
//  10494950
//  10489950
//  10489950
//  10484950
//  10474950
//  10474950
//  10469950
//  10469950
//  10464950
//  10464950
//  tbl_Sort_TD_0978_output #Row: 5093221
//  Filter_2: 9.215 ms tbl_SerializeFromObject_TD_3378_input: 1500000
//  JOIN_INNER_1: 429.269 ms tbl_Filter_TD_2351_output: 1272911 tbl_SerializeFromObject_TD_2598_input: 6001215
//  Sort_0: 519.48 ms tbl_JOIN_INNER_TD_1538_output: 5093221
//
//  Total execution time: 957 ms
//  Spark elapsed time: 5532.53ms
