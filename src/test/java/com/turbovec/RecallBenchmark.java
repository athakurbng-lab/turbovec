package com.turbovec;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public class RecallBenchmark {

    public static void main(String[] args) {
        int dim = 384;
        int bits = 4;
        int numVectors = 50000;
        int numQueries = 1000;
        int topK = 15;
        int calibSize = Math.min(5000, numVectors);

        System.out.println("Initializing TurboVec (dim=" + dim + ", bits=" + bits + ")...");
        long start = System.currentTimeMillis();
        TurboVec turbovec = new TurboVec(dim, bits);
        System.out.println("Initialization took " + (System.currentTimeMillis() - start) + " ms");

        System.out.println("Generating " + numVectors + " random vectors and " + numQueries + " queries...");
        Random rand = new Random(42);
        float[][] database = new float[numVectors][dim];
        for (int i = 0; i < numVectors; i++) {
            database[i] = randomUnitVector(dim, rand);
        }

        float[][] queries = new float[numQueries][dim];
        for (int i = 0; i < numQueries; i++) {
            queries[i] = randomUnitVector(dim, rand);
        }

        System.out.println("Fitting calibration on " + calibSize + " vectors...");
        float[][] calibBatch = new float[calibSize][dim];
        System.arraycopy(database, 0, calibBatch, 0, calibSize);
        start = System.currentTimeMillis();
        turbovec.fitCalibration(calibBatch);
        System.out.println("Calibration took " + (System.currentTimeMillis() - start) + " ms");

        System.out.println("Quantizing database...");
        start = System.currentTimeMillis();
        QuantizedVector[] quantizedDB = new QuantizedVector[numVectors];
        for (int i = 0; i < numVectors; i++) {
            quantizedDB[i] = turbovec.quantize(database[i]);
        }
        System.out.println("Quantization took " + (System.currentTimeMillis() - start) + " ms");

        System.out.println("Reconstructing database for scoring...");
        start = System.currentTimeMillis();
        float[][] approxDB = new float[numVectors][dim];
        for (int i = 0; i < numVectors; i++) {
            approxDB[i] = turbovec.convertToNormalSpace(quantizedDB[i].getPackedCodes(), quantizedDB[i].getNorm());
        }
        System.out.println("Reconstruction took " + (System.currentTimeMillis() - start) + " ms");

        System.out.println("Evaluating recall...");
        int recallAt1 = 0;
        int recallAt5 = 0;
        int recallAt10 = 0;
        int recallAt15 = 0;
        double totalOverlap = 0;

        start = System.currentTimeMillis();
        for (int q = 0; q < numQueries; q++) {
            float[] query = queries[q];

            // Exact scores
            DocScore[] exactScores = new DocScore[numVectors];
            for (int i = 0; i < numVectors; i++) {
                exactScores[i] = new DocScore(i, dotProduct(query, database[i]));
            }
            Arrays.sort(exactScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());
            
            int trueNN = exactScores[0].id;
            int[] trueTopK = new int[topK];
            for (int k = 0; k < topK; k++) {
                trueTopK[k] = exactScores[k].id;
            }

            // Approx scores
            DocScore[] approxScores = new DocScore[numVectors];
            for (int i = 0; i < numVectors; i++) {
                approxScores[i] = new DocScore(i, dotProduct(query, approxDB[i]));
            }
            Arrays.sort(approxScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());

            // Check if true NN is in top-K
            boolean foundIn1 = false;
            boolean foundIn5 = false;
            boolean foundIn10 = false;
            boolean foundIn15 = false;
            int overlap = 0;

            for (int k = 0; k < topK; k++) {
                int approxId = approxScores[k].id;
                if (k == 0 && approxId == trueNN) foundIn1 = true;
                if (k < 5 && approxId == trueNN) foundIn5 = true;
                if (k < 10 && approxId == trueNN) foundIn10 = true;
                if (k < 15 && approxId == trueNN) foundIn15 = true;

                for (int j = 0; j < topK; j++) {
                    if (approxId == trueTopK[j]) {
                        overlap++;
                        break;
                    }
                }
            }

            if (foundIn1) recallAt1++;
            if (foundIn5) recallAt5++;
            if (foundIn10) recallAt10++;
            if (foundIn15) recallAt15++;
            totalOverlap += overlap;
        }
        System.out.println("Evaluation took " + (System.currentTimeMillis() - start) + " ms");

        System.out.printf("Results over %d queries (Dim=%d, Bits=%d, N=%d):\n", numQueries, dim, bits, numVectors);
        System.out.printf("Recall@1:  %.2f%%\n", 100.0 * recallAt1 / numQueries);
        System.out.printf("Recall@5:  %.2f%%\n", 100.0 * recallAt5 / numQueries);
        System.out.printf("Recall@10: %.2f%%\n", 100.0 * recallAt10 / numQueries);
        System.out.printf("Recall@15: %.2f%%\n", 100.0 * recallAt15 / numQueries);
        System.out.printf("Overlap@15: %.2f%%\n", 100.0 * totalOverlap / (numQueries * topK));
    }

    private static float[] randomUnitVector(int dim, Random rand) {
        float[] v = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rand.nextGaussian();
            norm += v[i] * v[i];
        }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < dim; i++) {
            v[i] /= norm;
        }
        return v;
    }

    private static float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static class DocScore {
        int id;
        double score;
        DocScore(int id, double score) {
            this.id = id;
            this.score = score;
        }
    }
}
