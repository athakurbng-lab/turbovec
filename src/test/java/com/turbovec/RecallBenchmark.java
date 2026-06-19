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
        String queryFile = ""; // Link to query file (empty to generate random)
        String dataFile = "";  // Link to data file (empty to generate random)

        float[][] database;
        if (dataFile != null && !dataFile.trim().isEmpty()) {
            System.out.println("Loading data vectors from " + dataFile + "...");
            database = loadVectors(dataFile, dim);
            numVectors = database.length;
        } else {
            System.out.println("Generating " + numVectors + " random vectors...");
            Random rand = new Random(42);
            database = new float[numVectors][dim];
            for (int i = 0; i < numVectors; i++) {
                database[i] = randomUnitVector(dim, rand);
            }
        }

        float[][] queries;
        if (queryFile != null && !queryFile.trim().isEmpty()) {
            System.out.println("Loading query vectors from " + queryFile + "...");
            queries = loadVectors(queryFile, dim);
            numQueries = queries.length;
        } else {
            System.out.println("Generating " + numQueries + " random queries...");
            Random rand = new Random(43);
            queries = new float[numQueries][dim];
            for (int i = 0; i < numQueries; i++) {
                queries[i] = randomUnitVector(dim, rand);
            }
        }

        int calibSize = Math.min(5000, numVectors);

        int topK = 15;

        System.out.println("Initializing TurboVec (dim=" + dim + ", bits=" + bits + ")...");
        long start = System.currentTimeMillis();
        TurboVec turbovec = new TurboVec(dim, bits);
        System.out.println("Initialization took " + (System.currentTimeMillis() - start) + " ms");

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
        double totalRecall1 = 0;
        double totalRecall5 = 0;
        double totalRecall10 = 0;
        double totalRecall15 = 0;

        start = System.currentTimeMillis();
        for (int q = 0; q < numQueries; q++) {
            float[] query = queries[q];

            // Exact scores
            DocScore[] exactScores = new DocScore[numVectors];
            for (int i = 0; i < numVectors; i++) {
                exactScores[i] = new DocScore(i, dotProduct(query, database[i]));
            }
            Arrays.sort(exactScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());
            
            // Approx scores
            DocScore[] approxScores = new DocScore[numVectors];
            for (int i = 0; i < numVectors; i++) {
                approxScores[i] = new DocScore(i, dotProduct(query, approxDB[i]));
            }
            Arrays.sort(approxScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());

            // Check intersection overlap at different K values
            int overlap1 = computeIntersectionSize(exactScores, approxScores, 1);
            int overlap5 = computeIntersectionSize(exactScores, approxScores, 5);
            int overlap10 = computeIntersectionSize(exactScores, approxScores, 10);
            int overlap15 = computeIntersectionSize(exactScores, approxScores, 15);

            totalRecall1 += (double) overlap1 / 1.0;
            totalRecall5 += (double) overlap5 / 5.0;
            totalRecall10 += (double) overlap10 / 10.0;
            totalRecall15 += (double) overlap15 / 15.0;
        }
        System.out.println("Evaluation took " + (System.currentTimeMillis() - start) + " ms");

        System.out.printf("Results over %d queries (Dim=%d, Bits=%d, N=%d):\n", numQueries, dim, bits, numVectors);
        System.out.printf("Recall@1:  %.2f%%\n", 100.0 * totalRecall1 / numQueries);
        System.out.printf("Recall@5:  %.2f%%\n", 100.0 * totalRecall5 / numQueries);
        System.out.printf("Recall@10: %.2f%%\n", 100.0 * totalRecall10 / numQueries);
        System.out.printf("Recall@15: %.2f%%\n", 100.0 * totalRecall15 / numQueries);
    }

    private static int computeIntersectionSize(DocScore[] exactScores, DocScore[] approxScores, int k) {
        int overlap = 0;
        for (int i = 0; i < k; i++) {
            int approxId = approxScores[i].id;
            for (int j = 0; j < k; j++) {
                if (approxId == exactScores[j].id) {
                    overlap++;
                    break;
                }
            }
        }
        return overlap;
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

    private static float[][] loadVectors(String filePath, int expectedDim) {
        try {
            java.util.List<String> lines = java.nio.file.Files.readAllLines(java.nio.file.Paths.get(filePath));
            float[][] vectors = new float[lines.size()][expectedDim];
            for (int i = 0; i < lines.size(); i++) {
                String[] parts = lines.get(i).trim().split("\\s+");
                for (int j = 0; j < expectedDim; j++) {
                    vectors[i][j] = Float.parseFloat(parts[j]);
                }
            }
            return vectors;
        } catch (Exception e) {
            throw new RuntimeException("Error reading file: " + filePath, e);
        }
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
