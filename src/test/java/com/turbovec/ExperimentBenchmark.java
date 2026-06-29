package com.turbovec;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import java.util.stream.IntStream;

public class ExperimentBenchmark {

    public static void main(String[] args) {
        int dim = 384;
        int numVectors = 50000;
        int numQueries = 1000;

        float[][] database = loadVectorsFromFile("database.txt", dim);
        if (database == null) {
            long dbSeed = Long.parseLong(System.getProperty("dbSeed", "42"));
            System.out.println("No database.txt found. Generating " + numVectors + " random database vectors with seed " + dbSeed + "...");
            Random dbRand = new Random(dbSeed);
            database = new float[numVectors][dim];
            for (int i = 0; i < numVectors; i++) {
                database[i] = randomUnitVector(dim, dbRand);
            }
        } else {
            numVectors = database.length;
            System.out.println("Loaded " + numVectors + " database vectors from file.");
        }

        float[][] queries = loadVectorsFromFile("queries.txt", dim);
        if (queries == null) {
            long qSeed = Long.parseLong(System.getProperty("qSeed", "43"));
            System.out.println("No queries.txt found. Generating " + numQueries + " random queries with seed " + qSeed + "...");
            Random qRand = new Random(qSeed);
            queries = new float[numQueries][dim];
            for (int i = 0; i < numQueries; i++) {
                queries[i] = randomUnitVector(dim, qRand);
            }
        } else {
            numQueries = queries.length;
            System.out.println("Loaded " + numQueries + " queries from file.");
        }

        // Exact scores since database and query remain unchanged
        System.out.println("Computing exact ground truth...");
        int[][] exactTopIds = new int[numQueries][15];
        final float[][] finalDatabase = database;
        final float[][] finalQueries = queries;
        final int finalNumVectors = numVectors;
        final int finalDim = dim;
        
        IntStream.range(0, numQueries).parallel().forEach(q -> {
            DocScore[] scores = new DocScore[finalNumVectors];
            for (int i = 0; i < finalNumVectors; i++) {
                float dot = 0;
                for (int d = 0; d < finalDim; d++) {
                    dot += finalQueries[q][d] * finalDatabase[i][d];
                }
                scores[i] = new DocScore(i, dot);
            }
            Arrays.sort(scores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());
            for (int k = 0; k < 15; k++) {
                exactTopIds[q][k] = scores[k].id;
            }
        });

        // Configuration combinations
        // 1. 4-bit, 25%, 0, true, Asymmetric
        // 2. 8-bit, 50%, 100, false, Asymmetric
        // 3. 16-bit, 10%, 0, false, Asymmetric
        
        System.out.printf("\n%-6s | %-6s | %-5s | %-6s | %-12s | %-8s | %-8s | %-8s\n",
                "Bits", "Calib%", "Seed", "Row1st", "Mode", "Recall@5", "Recall@10", "Recall@15");
        System.out.println("-".repeat(80));

        runConfig(database, queries, exactTopIds, 4, 0.25, 0L, true);
        runConfig(database, queries, exactTopIds, 8, 0.50, 100L, false);
        runConfig(database, queries, exactTopIds, 16, 0.10, 0L, false);
    }

    private static void runConfig(float[][] database, float[][] queries, int[][] exactTopIds, int bits, double calibPercent, long seed, boolean fillRowFirst) {
        int dim = database[0].length;
        int numVectors = database.length;
        int calibSize = (int) (numVectors * calibPercent);
        float[][] calibBatch = new float[calibSize][dim];
        System.arraycopy(database, 0, calibBatch, 0, calibSize);

        TurboVec turbovec = new TurboVec(dim, bits, seed, fillRowFirst);
        turbovec.fitCalibration(calibBatch);

        QuantizedVector[] quantizedDB = turbovec.quantizeBatch(database);

        evaluate(turbovec, queries, quantizedDB, exactTopIds, "Asymmetric", bits, calibPercent, seed, fillRowFirst, false);
        evaluate(turbovec, queries, quantizedDB, exactTopIds, "Asymm+QJL", bits, calibPercent, seed, fillRowFirst, true);
    }

    private static void evaluate(TurboVec turbovec, float[][] queries, QuantizedVector[] quantizedDB, int[][] exactTopIds,
                                 String mode, int bits, double calibPercent, long seed, boolean fillRowFirst, boolean useQjl) {
        int numQueries = queries.length;
        int numVectors = quantizedDB.length;
        int dim = queries[0].length;

        int[][] unpackedDB = new int[numVectors][];
        if (bits <= 8) {
            IntStream.range(0, numVectors).parallel().forEach(i -> {
                unpackedDB[i] = turbovec.unpack(quantizedDB[i].getPackedCodes());
            });
        }

        int[] recall5 = new int[1];
        int[] recall10 = new int[1];
        int[] recall15 = new int[1];

        final int finalDim2 = dim;

        IntStream.range(0, numQueries).parallel().forEach(q -> {
            DocScore[] approxScores = new DocScore[numVectors];
            
            if (bits <= 8) {
                float[] rotatedQ = turbovec.rotateQuery(queries[q]);
                float[][] lut = turbovec.precomputeAsymmetricLUT(rotatedQ);

                for (int i = 0; i < numVectors; i++) {
                    float score;
                    if (useQjl) {
                        score = turbovec.asymmetricDotProductLUTWithQJL(lut, unpackedDB[i], quantizedDB[i].getQjlBits(), quantizedDB[i].getQjlScale(), rotatedQ);
                    } else {
                        score = turbovec.asymmetricDotProductLUT(lut, unpackedDB[i]);
                    }
                    approxScores[i] = new DocScore(i, score);
                }
            } else {
                for (int i = 0; i < numVectors; i++) {
                    float[] reconstructed = turbovec.convertToNormalSpace(quantizedDB[i].getPackedCodes());
                    
                    if (useQjl) {
                        byte[] qjlBits = quantizedDB[i].getQjlBits();
                        float qjlScale = quantizedDB[i].getQjlScale();
                        float[] rotatedQ = turbovec.rotateQuery(queries[q]);
                        float qjlDot = 0;
                        int bytesPerPlane = finalDim2 / 8;
                        for (int by = 0; by < bytesPerPlane; by++) {
                            byte b = qjlBits[by];
                            for (int bit = 0; bit < 8; bit++) {
                                int d = by * 8 + bit;
                                if ((b & (1 << (7 - bit))) != 0) {
                                    qjlDot += rotatedQ[d];
                                } else {
                                    qjlDot -= rotatedQ[d];
                                }
                            }
                        }
                        
                        // We also need to add the QJL dot to the dot product. But QJL uses rotated query and rotated DB residuals.
                        // Actually, <q, r> = <q_rot, r_rot>. 
                        // Our QJL bits encode r_rot. So we just add qjlScale * qjlDot to the normal space dot product.
                        
                        float dot = 0;
                        for (int d = 0; d < finalDim2; d++) {
                            dot += queries[q][d] * reconstructed[d];
                        }
                        approxScores[i] = new DocScore(i, dot + qjlScale * qjlDot);
                    } else {
                        float dot = 0;
                        for (int d = 0; d < finalDim2; d++) {
                            dot += queries[q][d] * reconstructed[d];
                        }
                        approxScores[i] = new DocScore(i, dot);
                    }
                }
            }
            
            Arrays.sort(approxScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());

            int overlap5 = computeIntersectionSize(exactTopIds[q], approxScores, 5);
            int overlap10 = computeIntersectionSize(exactTopIds[q], approxScores, 10);
            int overlap15 = computeIntersectionSize(exactTopIds[q], approxScores, 15);

            synchronized (recall5) {
                recall5[0] += overlap5;
                recall10[0] += overlap10;
                recall15[0] += overlap15;
            }
        });

        System.out.printf("%-6d | %-6.2f | %-5d | %-6b | %-12s | %-7.2f%% | %-7.2f%% | %-7.2f%%\n",
                bits, calibPercent * 100, seed, fillRowFirst, mode,
                100.0 * recall5[0] / (numQueries * 5.0), 100.0 * recall10[0] / (numQueries * 10.0), 100.0 * recall15[0] / (numQueries * 15.0));
    }

    private static void evaluateSymmetric(TurboVec turbovec, QuantizedVector[] quantizedQueries, QuantizedVector[] quantizedDB, int[][] exactTopIds,
                                          String mode, int bits, double calibPercent, long seed, boolean fillRowFirst) {
        int numQueries = quantizedQueries.length;
        int numVectors = quantizedDB.length;
        int dim = turbovec.getDim();
        double r5 = 0, r10 = 0, r15 = 0;

        int[][] unpackedDB = new int[numVectors][dim];
        for (int i = 0; i < numVectors; i++) {
            turbovec.unpack(quantizedDB[i].getPackedCodes(), unpackedDB[i]);
        }

        for (int q = 0; q < numQueries; q++) {
            DocScore[] approxScores = new DocScore[numVectors];
            
            if (bits <= 8) {
                // To do symmetric LUT, we unpack the query and convert to normal space (this is exactly what Symmetric scoring is)
                float[] reconQ = turbovec.convertToNormalSpace(quantizedQueries[q].getPackedCodes());
                float[] rotQ = turbovec.rotateQuery(reconQ);
                float[][] lut = turbovec.precomputeAsymmetricLUT(rotQ);
                for (int i = 0; i < numVectors; i++) {
                    approxScores[i] = new DocScore(i, turbovec.asymmetricDotProductLUT(lut, unpackedDB[i]));
                }
            } else {
                for (int i = 0; i < numVectors; i++) {
                    approxScores[i] = new DocScore(i, turbovec.symmetricDotProduct(quantizedQueries[q], quantizedDB[i]));
                }
            }
            
            Arrays.sort(approxScores, Comparator.comparingDouble((DocScore s) -> s.score).reversed());

            r5 += computeIntersectionSize(exactTopIds[q], approxScores, 5) / 5.0;
            r10 += computeIntersectionSize(exactTopIds[q], approxScores, 10) / 10.0;
            r15 += computeIntersectionSize(exactTopIds[q], approxScores, 15) / 15.0;
        }

        System.out.printf("%-6d | %-6.2f | %-5d | %-6b | %-12s | %-7.2f%% | %-7.2f%% | %-7.2f%%\n",
                bits, calibPercent * 100, seed, fillRowFirst, mode,
                100.0 * r5 / numQueries, 100.0 * r10 / numQueries, 100.0 * r15 / numQueries);
    }

    private static float[][] loadVectorsFromFile(String filename, int expectedDim) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            List<float[]> vectors = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("[,\\s]+");
                if (parts.length != expectedDim) {
                    System.out.println("Warning: Skipping line in " + filename + " because it doesn't have " + expectedDim + " dimensions.");
                    continue;
                }
                float[] v = new float[expectedDim];
                float norm = 0;
                for (int i = 0; i < expectedDim; i++) {
                    v[i] = Float.parseFloat(parts[i]);
                    norm += v[i] * v[i];
                }
                // Normalize loaded vectors
                if (norm > 0) {
                    norm = (float) Math.sqrt(norm);
                    for (int i = 0; i < expectedDim; i++) {
                        v[i] /= norm;
                    }
                }
                vectors.add(v);
            }
            if (vectors.isEmpty()) return null;
            return vectors.toArray(new float[0][]);
        } catch (IOException e) {
            return null; // Return null to fallback to random generation
        }
    }

    private static int computeIntersectionSize(int[] exactIds, DocScore[] approxScores, int k) {
        int overlap = 0;
        for (int i = 0; i < k; i++) {
            int approxId = approxScores[i].id;
            for (int j = 0; j < k; j++) {
                if (approxId == exactIds[j]) {
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
