import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.io.IOException;
import java.util.*;

public class G29HW1 {

    /** Parameters :
     * args[0] : K : number of partitions ;
     * args[1] : T : number of best products to display ;
     * args[2] : name of the input file.
    */
    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true).setAppName("HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");


        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read number of best products to display
        int T = Integer.parseInt(args[1]);

        // Step 1 : Read input file and subdivide it into K random partitions
        JavaRDD<String> RawData = sc.textFile(args[2]).repartition(K).cache();


        // Step 2
        JavaPairRDD<String, Double> normalizedRatings =
                RawData.mapToPair((document) -> // First map phase
                {
                    String[] tokens = document.split(",");

                    // We build pairs : (UserID, (ProductID, Rating)) for each line.
                    return new Tuple2<>(tokens[1], new Tuple2<>(tokens[0], Double.parseDouble(tokens[2])));

                })
                        .groupByKey() // First reduce phase
                        .flatMapToPair((element) -> { // Second reduce phase
                            int i = 0;
                            double mean = 0;

                            // For each (ProductID, Rating) pair
                            for (Tuple2<String, Double> t : element._2())
                            {
                                mean += Double.parseDouble(t._2().toString()) ;
                                i++;
                            }
                            mean = mean / i;

                            ArrayList<Tuple2<String, Double>> normProducts = new ArrayList<>();

                            for (Tuple2<String, Double> t : element._2())
                            {
                                Double normMean = t._2()-mean;
                                normProducts.add(new Tuple2<>(t._1(), normMean));
                            }

                            return  normProducts.iterator();
                        });

        // Step 3 : only a reduce phase
        JavaPairRDD<String, Double> maxNormRatings = normalizedRatings.reduceByKey(
                (v1, v2) -> Math.max(v1, v2)

        );


        // Step 4 : we swap keys and values, then we use sortByKeys, and we swap again.
        JavaPairRDD<Double, String> swap = maxNormRatings.mapToPair(x -> x.swap());
        JavaPairRDD<Double, String> sorted = swap.sortByKey(false);
        JavaPairRDD<String, Double> ordered = sorted.mapToPair(x -> x.swap());

        for (int i = 0; i < T; i++)
        {
            System.out.println("Product " + ordered.keys().take(10).get(i).toString() + " maxNormRating " + ordered.values().take(10).get(i).toString());

        }

    }


}

