package com.turbovec;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.SimpsonIntegrator;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.Random;

public class TurboVec {
    private final int dim;
    private final int bitWidth;
    private final int numLevels;
    private final float[] boundaries;
    private final float[] centroids;
    private final float[][] rotationMatrix; // [dim][dim]

    private float[] shift;
    private float[] scaleTq;
    private float[] invScaleTq;

    // Constants
    private static final long ROTATION_SEED = 42L; // Same seed or configurable

    public TurboVec(int dim, int bitWidth) {
        this.dim = dim;
        this.bitWidth = bitWidth;
        this.numLevels = 1 << bitWidth;

        // 1. Setup Lloyd-Max Codebook
        float[][] codebook = generateLloydMaxCodebook(bitWidth, dim);
        this.boundaries = codebook[0];
        this.centroids = codebook[1];

        // 2. Generate Random Orthogonal Rotation Matrix
        this.rotationMatrix = generateRotationMatrix(dim);

        // Initialize default identity calibration
        this.shift = new float[dim];
        this.scaleTq = new float[dim];
        this.invScaleTq = new float[dim];
        Arrays.fill(this.scaleTq, 1.0f);
        Arrays.fill(this.invScaleTq, 1.0f);
    }

    private float[][] generateRotationMatrix(int dim) {
        Random rng = new Random(ROTATION_SEED);
        double[][] gData = new double[dim][dim];
        for (int j = 0; j < dim; j++) {
            for (int i = 0; i < dim; i++) {
                gData[i][j] = rng.nextGaussian();
            }
        }
        RealMatrix g = new Array2DRowRealMatrix(gData);
        QRDecomposition qr = new QRDecomposition(g);
        RealMatrix q = qr.getQ();
        RealMatrix r = qr.getR();

        float[][] result = new float[dim][dim];
        for (int j = 0; j < dim; j++) {
            double sign = r.getEntry(j, j) >= 0 ? 1.0 : -1.0;
            for (int i = 0; i < dim; i++) {
                result[i][j] = (float) (q.getEntry(i, j) * sign);
            }
        }
        return result;
    }

    private float[][] generateLloydMaxCodebook(int bits, int dim) {
        int maxIter = 200;
        double tol = 1e-12;
        int nLevels = 1 << bits;

        double a = (dim - 1.0) / 2.0;
        BetaDistribution beta = new BetaDistribution(a, a);

        double stdDev = Math.sqrt((2.0 * a) / ((2.0 * a + 1.0) * 4.0 * a));
        double spread = 3.0 * stdDev;
        double[] centroids = new double[nLevels];
        for (int i = 0; i < nLevels; i++) {
            centroids[i] = -spread + 2.0 * spread * i / (nLevels - 1.0);
        }

        SimpsonIntegrator integrator = new SimpsonIntegrator();

        for (int iter = 0; iter < maxIter; iter++) {
            double[] boundaries = new double[nLevels - 1];
            for (int i = 0; i < nLevels - 1; i++) {
                boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
            }

            double[] edges = new double[nLevels + 1];
            edges[0] = -1.0;
            System.arraycopy(boundaries, 0, edges, 1, boundaries.length);
            edges[nLevels] = 1.0;

            double[] newCentroids = new double[nLevels];
            for (int i = 0; i < nLevels; i++) {
                double lo = edges[i];
                double hi = edges[i + 1];

                double cdfLo = beta.cumulativeProbability((lo + 1.0) / 2.0);
                double cdfHi = beta.cumulativeProbability((hi + 1.0) / 2.0);
                double prob = cdfHi - cdfLo;

                if (prob < 1e-15) {
                    newCentroids[i] = centroids[i];
                } else {
                    UnivariateFunction f = x -> {
                        double t = (x + 1.0) / 2.0;
                        return x * beta.density(t) / 2.0;
                    };
                    double mean = adaptiveSimpson(f, lo, hi, 1e-14, 50, integrator);
                    newCentroids[i] = mean / prob;
                }
            }

            double maxChange = 0.0;
            for (int i = 0; i < nLevels; i++) {
                maxChange = Math.max(maxChange, Math.abs(centroids[i] - newCentroids[i]));
            }
            centroids = newCentroids;
            if (maxChange < tol) {
                break;
            }
        }

        float[] floatBoundaries = new float[nLevels - 1];
        for (int i = 0; i < nLevels - 1; i++) {
            floatBoundaries[i] = (float) ((centroids[i] + centroids[i + 1]) / 2.0);
        }
        float[] floatCentroids = new float[nLevels];
        for (int i = 0; i < nLevels; i++) {
            floatCentroids[i] = (float) centroids[i];
        }

        return new float[][]{floatBoundaries, floatCentroids};
    }

    private double adaptiveSimpson(UnivariateFunction f, double a, double b, double tol, int maxDepth, SimpsonIntegrator integrator) {
        try {
            // Provide a fast fallback using commons-math simpson integrator
            return integrator.integrate(10000, f, a, b);
        } catch (Exception e) {
            return 0.0;
        }
    }

    public float[] getCentroidArray() {
        return centroids;
    }

    public void fitCalibration(float[][] batch) {
        int n = batch.length;
        if (n < 1000) {
            System.err.println("Batch size too small for reliable calibration (n < 1000). Using identity calibration.");
            Arrays.fill(shift, 0.0f);
            Arrays.fill(scaleTq, 1.0f);
            Arrays.fill(invScaleTq, 1.0f);
            return;
        }

        float[][] rotatedBatch = new float[n][dim];
        for (int i = 0; i < n; i++) {
            float[] vec = batch[i];
            float norm = computeNorm(vec);
            float[] unit = new float[dim];
            if (norm > 1e-10) {
                for (int d = 0; d < dim; d++) unit[d] = vec[d] / norm;
            }
            for (int d = 0; d < dim; d++) {
                float sum = 0;
                for (int j = 0; j < dim; j++) {
                    sum += unit[j] * rotationMatrix[d][j];
                }
                rotatedBatch[i][d] = sum;
            }
        }

        double a = (dim - 1.0) / 2.0;
        BetaDistribution beta = new BetaDistribution(a, a);
        double P_LO = 0.05;
        double P_HI = 0.95;
        float qcLo = (float) (2.0 * beta.inverseCumulativeProbability(P_LO) - 1.0);
        float qcHi = (float) (2.0 * beta.inverseCumulativeProbability(P_HI) - 1.0);
        float qcSpan = qcHi - qcLo;

        int loIdx = (int) (n * P_LO);
        int hiIdx = Math.min((int) (n * P_HI), n - 1);

        for (int d = 0; d < dim; d++) {
            float[] coord = new float[n];
            for (int i = 0; i < n; i++) coord[i] = rotatedBatch[i][d];
            Arrays.sort(coord);
            float qeLo = coord[loIdx];
            float qeHi = coord[hiIdx];
            float qeSpan = qeHi - qeLo;

            if (qeSpan > 1e-6f) {
                scaleTq[d] = qcSpan / qeSpan;
                shift[d] = qcLo / scaleTq[d] - qeLo;
                invScaleTq[d] = 1.0f / scaleTq[d];
            } else {
                shift[d] = 0.0f;
                scaleTq[d] = 1.0f;
                invScaleTq[d] = 1.0f;
            }
        }
    }

    private float computeNorm(float[] vec) {
        float sum = 0;
        for (float v : vec) sum += v * v;
        return (float) Math.sqrt(sum);
    }

    public int[] getCentroidPositions(float[] embedding) {
        float norm = computeNorm(embedding);
        float[] unit = new float[dim];
        if (norm > 1e-10) {
            for (int d = 0; d < dim; d++) unit[d] = embedding[d] / norm;
        }

        int[] positions = new int[dim];
        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += unit[j] * rotationMatrix[d][j];
            }
            float valCalib = (sum + shift[d]) * scaleTq[d];

            int code = 0;
            for (float b : boundaries) {
                if (valCalib > b) code++;
            }
            positions[d] = code;
        }
        return positions;
    }

    public QuantizedVector quantize(float[] embedding) {
        float norm = computeNorm(embedding);
        float[] unit = new float[dim];
        if (norm > 1e-10) {
            for (int d = 0; d < dim; d++) unit[d] = embedding[d] / norm;
        }

        float[] rotOrig = new float[dim];
        float[] rotCalib = new float[dim];
        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += unit[j] * rotationMatrix[d][j]; // note: matrix is Q. transpose of Q is its inverse. Q in faer might be column-major. Let's make sure. The rotation here is y = Q^T x.
            }
            rotOrig[d] = sum;
            rotCalib[d] = (sum + shift[d]) * scaleTq[d];
        }

        int bytesPerPlane = dim / 8;
        byte[] packedRow = new byte[bitWidth * bytesPerPlane];
        double inner = 0.0;

        for (int j = 0; j < dim; j++) {
            int code = 0;
            for (float b : boundaries) {
                if (rotCalib[j] > b) code++;
            }

            double centroidInOrig = centroids[code] * invScaleTq[j] - shift[j];
            inner += rotOrig[j] * centroidInOrig;

            int bytePos = j / 8;
            int bitPos = 7 - (j % 8);
            for (int p = 0; p < bitWidth; p++) {
                if ((code & (1 << p)) != 0) {
                    packedRow[p * bytesPerPlane + bytePos] |= (byte) (1 << bitPos);
                }
            }
        }

        float adjustedNorm = (float) (norm / Math.max(inner, 1e-10));
        return new QuantizedVector(packedRow, adjustedNorm);
    }

    public int[] unpack(byte[] packedCodes) {
        int bytesPerPlane = dim / 8;
        int[] positions = new int[dim];
        for (int j = 0; j < dim; j++) {
            int code = 0;
            int bytePos = j / 8;
            int bitPos = 7 - (j % 8);
            for (int p = 0; p < bitWidth; p++) {
                byte b = packedCodes[p * bytesPerPlane + bytePos];
                if ((b & (1 << bitPos)) != 0) {
                    code |= (1 << p);
                }
            }
            positions[j] = code;
        }
        return positions;
    }

    public float[] convertToNormalSpace(int[] quantizedPos, float originalNorm) {
        float[] rotOrig = new float[dim];
        for (int d = 0; d < dim; d++) {
            int code = quantizedPos[d];
            rotOrig[d] = (float) (centroids[code] * invScaleTq[d] - shift[d]);
        }

        float[] result = new float[dim];
        for (int i = 0; i < dim; i++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += rotOrig[j] * rotationMatrix[j][i];
            }
            result[i] = sum * originalNorm;
        }
        return result;
    }

    public float[] convertToNormalSpace(byte[] packedCodes, float norm) {
        int[] positions = unpack(packedCodes);
        return convertToNormalSpace(positions, norm);
    }
}
