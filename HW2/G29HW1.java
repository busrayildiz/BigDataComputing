import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.io.IOException;
import java.util.*;



public class G29HW1 {

    public static Tuple2<Vector, Integer> strToTuple (String str){
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = Vectors.dense(data);
        Integer cluster = Integer.valueOf(tokens[tokens.length-1]);
        Tuple2<Vector, Integer> pair = new Tuple2<>(point, cluster);
        return pair;
    }

    /* Returns the sum of squared euclidean distances between the pair and the elements of a sample
     * which have the id "clusterID".
     */
    public static double sumOfDistances(Tuple2<Vector, Integer> pair,
                                        List<Tuple2<Vector, Integer>> clusteringSample,
                                        int clusterID)
    {
        double sum = 0.0;
        for (int i = 0 ; i < clusteringSample.size() ; i++)
        {
            if (clusteringSample.get(i)._2() == clusterID)
            {
                sum += Vectors.sqdist(pair._1(), clusteringSample.get(i)._1());
            }
        }
        return sum;
    }

    /** Parameters :
     * args[0] : input_file : path of the input file ;
     * args[1] : k : number of clusters
     * args[2] : t : expected sample size.
    */
    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true).setAppName("HW2");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read number of clusters
        int k = Integer.parseInt(args[1]);

        // Read expected size per sample
        int t = Integer.parseInt(args[2]);

        // Step 1 : Read input file and subdivide it into K random partitions
        // Pairs : ((Xi, Yi), ClusterNumber)
        JavaPairRDD<Vector,Integer> fullClustering = sc.textFile(args[0]).repartition(4)
                .mapToPair(x -> strToTuple(x)).cache();

        // Step 2 : Computing the clusters sizes.
        ArrayList<Long> sharedClusterSizes = new ArrayList<Long>(fullClustering.map(
                x->(x._2())).countByValue().values()
        );

        // System.out.println(sharedClusterSizes.toString());

        // Step 3 : Take a sample of the main set
        Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = sc.broadcast(
                fullClustering.filter(
                x -> {
                    /* sharedClusterSizes[x._2()] = size of the cluster in which x is */
                    Long c = sharedClusterSizes.get(x._2());
                    double probability = (double) t / (double) c;
                    return (probability >= Math.random());
                }
            ).collect()
        );

        // Step 4 : approximating the Silhouette coefficient
        long start = System.currentTimeMillis();

        double approxSilhFull = fullClustering.map(
                p -> {
                    // ID of the cluster in which p is (same convention as the slides given in the assignment)
                    int i = p._2();

                    long c_i = sharedClusterSizes.get(i);
                    double a_p = (1.0 / Math.min(t, c_i)) * sumOfDistances(p, clusteringSample.value(), i);

                    // Computing b_p :
                    // Initialize the minimum first
                    double b_p = 0.0;
                    if (i >= 1) // We take the term with j = 0 (as shown on the slides)
                    {
                        long c_0 = sharedClusterSizes.get(0);
                        b_p = (1.0 / Math.min(t, c_0)) * sumOfDistances(p, clusteringSample.value(), 0);
                    }
                    else // i == 0 : take the term with j = 1
                    {
                        long c_1 = sharedClusterSizes.get(1);
                        b_p = (1.0 / Math.min(t, c_1)) * sumOfDistances(p, clusteringSample.value(), 1);
                    }

                    // Computing the minimum we want for b_p
                    for (int j = 0 ; j < sharedClusterSizes.size() ; j++)
                    {
                        if (j != i)
                        {
                            long c_j = sharedClusterSizes.get(j);
                            double term_j = (1.0 / Math.min(t, c_j)) * sumOfDistances(p, clusteringSample.value(), j);
                            if (term_j < b_p)
                            {
                                b_p = term_j;
                            }
                        }
                    }

                    return ((b_p - a_p) / Math.max(a_p, b_p)); // silhouette coeff. for the pair p

                }
        ).reduce(Double::sum) / fullClustering.count(); // We calculate the mean of s_p values
        long end = System.currentTimeMillis();

        System.out.println("Value of approxSilhFull = " + approxSilhFull);
        System.out.println("Time to compute approxSilhFull = " + (end - start) + " ms");

        // Step 5 : exact silhouette coefficient of the sample
        start = System.currentTimeMillis();
        long[] sampleClusterSizes = new long[k]; // We need the sizes of the subclusters in the sample
        for (int i = 0 ; i < clusteringSample.value().size() ; i++)
        {
            int clusterID = clusteringSample.value().get(i)._2();
            sampleClusterSizes[clusterID] += 1;
        }

        //System.out.println(Arrays.toString(sampleClusterSizes));

        double[] array_sp = new double[clusteringSample.value().size()];
        // The only difference is that we go from the cluster sample to select p :
        for (int idx_pair = 0 ; idx_pair < clusteringSample.value().size() ; idx_pair++)
        {
            Tuple2<Vector, Integer> p = clusteringSample.value().get(idx_pair);
            int i = p._2(); // Cluster ID
            long c_i = sampleClusterSizes[p._2()];

            double a_p = (1.0 / Math.min(t, c_i)) * sumOfDistances(p, clusteringSample.value(), p._2());

            // Computing b_p :
            // Initialize the minimum first
            double b_p = 0.0;
            if (i >= 1) // We take the term with j = 0 (as shown on the slides)
            {
                long c_0 = sampleClusterSizes[0];
                b_p = (1.0 / Math.min(t, c_0)) * sumOfDistances(p, clusteringSample.value(), 0);
            }
            else
            {
                // i = 0 so we initialize b_p with the term j=1
                long c_1 = sampleClusterSizes[1]; // Size of cluster 1

                b_p = (1.0 / Math.min(t, c_1)) * sumOfDistances(p, clusteringSample.value(), 1);
            }

            for (int j = 0 ; j < sharedClusterSizes.size() ; j++)
            {
                if (j != i)
                {
                    long c_j = sampleClusterSizes[j];

                    double term_j = (1.0 / Math.min(t, c_j)) * sumOfDistances(p, clusteringSample.value(), j);

                    if (term_j < b_p)
                    {
                        b_p = term_j;
                    }
                }
            }

            // System.out.println("bp : " + Double.toString(b_p) + ", ap : " + Double.toString(a_p));
            array_sp[idx_pair] = (b_p - a_p) / Math.max(a_p, b_p);

        }

        // Computing the mean
        double exactSilhSample = 0.0;
        for (double sp : array_sp)
        {
            exactSilhSample += sp;
        }

        exactSilhSample = exactSilhSample / clusteringSample.value().size();

        end = System.currentTimeMillis();


        System.out.println("Value of exactSilhSample = " + exactSilhSample);
        System.out.println("Time to compute exactSilhSample = " + (end - start) + " ms");

    }


}

