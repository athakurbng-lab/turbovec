package com.turbovec;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.Random;

/**
 * Java implementation of the TurboVec quantization primitives.
 *
 * <p>The validation, codebook, packing, TQ+ calibration, and raw-query scoring
 * semantics are aligned with turbovec-main. Packed bytes are not guaranteed to
 * be binary-compatible with turbovec-main yet because Rust uses
 * ChaCha8Rng + StandardNormal for its deterministic rotation matrix, while this
 * Java implementation still uses java.util.Random with the same seed.</p>
 */
public class TurboVec {
    public static final int MAX_DIM = 65_536;
    private static final float MAX_INPUT_MAGNITUDE = 1e16f;
    private static final int TQPLUS_MIN_SAMPLES = 1000;
    private static final double TQPLUS_P_LO = 0.05;
    private static final double TQPLUS_P_HI = 0.95;

    private final int dim;
    private final int bitWidth;
    private final float[] boundaries;
    private final float[] centroids;
    private final float[][] rotationMatrix; // [dim][dim]

    private float[] shift;
    private float[] scaleTq;
    private float[] invScaleTq;
    private boolean hasQuantizedVectors;

    public TurboVec(int dim, int bitWidth) {
        this(dim, bitWidth, 42L, false);
    }

    public TurboVec(int dim, int bitWidth, long seed, boolean fillRowFirst) {
        validateConstructorArgs(dim, bitWidth);
        this.dim = dim;
        this.bitWidth = bitWidth;

        float[][] codebook = generateLloydMaxCodebook(bitWidth, dim);
        this.boundaries = codebook[0];
        this.centroids = codebook[1];
        this.rotationMatrix = generateRotationMatrix(dim, seed, fillRowFirst);

        this.shift = new float[dim];
        this.scaleTq = new float[dim];
        this.invScaleTq = new float[dim];
        Arrays.fill(this.scaleTq, 1.0f);
        Arrays.fill(this.invScaleTq, 1.0f);
    }

    private static void validateConstructorArgs(int dim, int bitWidth) {
        if (bitWidth < 2 || bitWidth > 16) {
            throw new IllegalArgumentException("bitWidth must be between 2 and 16");
        }
        if (dim <= 0 || dim % 8 != 0) {
            throw new IllegalArgumentException("dim must be positive and divisible by 8");
        }
        if (dim > MAX_DIM) {
            throw new IllegalArgumentException("dim must be <= " + MAX_DIM);
        }
    }

    public int getDim() {
        return dim;
    }

    private float[][] generateRotationMatrix(int dim, long seed, boolean fillRowFirst) {
        Random rng = new Random(seed);
        double[][] gData = new double[dim][dim];
        if (fillRowFirst) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    gData[i][j] = rng.nextGaussian();
                }
            }
        } else {
            for (int j = 0; j < dim; j++) {
                for (int i = 0; i < dim; i++) {
                    gData[i][j] = rng.nextGaussian();
                }
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

    static float[][] generateLloydMaxCodebook(int bits, int dim) {
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
                    double mean = adaptiveSimpson(f, lo, hi, 1e-14, 50);
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

    private static double adaptiveSimpson(UnivariateFunction f, double a, double b, double tol, int maxDepth) {
        double mid = (a + b) / 2.0;
        double fa = f.value(a);
        double fb = f.value(b);
        double fm = f.value(mid);
        double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
        double result = adaptiveSimpsonRec(f, a, b, fa, fb, fm, whole, tol, maxDepth);
        if (!Double.isFinite(result)) {
            throw new IllegalStateException("Lloyd-Max codebook integration produced a non-finite result");
        }
        return result;
    }

    private static double adaptiveSimpsonRec(
            UnivariateFunction f,
            double a,
            double b,
            double fa,
            double fb,
            double fm,
            double whole,
            double tol,
            int depth
    ) {
        double mid = (a + b) / 2.0;
        double m1 = (a + mid) / 2.0;
        double m2 = (mid + b) / 2.0;
        double fm1 = f.value(m1);
        double fm2 = f.value(m2);
        double left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
        double right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
        double refined = left + right;
        if (depth == 0 || Math.abs(refined - whole) < 15.0 * tol) {
            return refined + (refined - whole) / 15.0;
        }
        return adaptiveSimpsonRec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
                + adaptiveSimpsonRec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1);
    }

    public float[] getCentroidArray() {
        return Arrays.copyOf(centroids, centroids.length);
    }

    public float[] getBoundaryArray() {
        return Arrays.copyOf(boundaries, boundaries.length);
    }

    public void fitCalibration(float[][] batch) {
        if (hasQuantizedVectors) {
            throw new IllegalStateException("calibration cannot be refit after vectors have been quantized");
        }
        validateBatch(batch);
        int n = batch.length;
        if (n < TQPLUS_MIN_SAMPLES) {
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
        float qcLo = (float) (2.0 * beta.inverseCumulativeProbability(TQPLUS_P_LO) - 1.0);
        float qcHi = (float) (2.0 * beta.inverseCumulativeProbability(TQPLUS_P_HI) - 1.0);
        float qcSpan = qcHi - qcLo;

        int loIdx = (int) (n * TQPLUS_P_LO);
        int hiIdx = Math.min((int) (n * TQPLUS_P_HI), n - 1);

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

    private void validateBatch(float[][] batch) {
        if (batch == null) {
            throw new IllegalArgumentException("batch must not be null");
        }
        for (int i = 0; i < batch.length; i++) {
            validateVector(batch[i], "batch[" + i + "]");
        }
    }

    private void validateVector(float[] vec, String name) {
        if (vec == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        if (vec.length != dim) {
            throw new IllegalArgumentException(name + " length must equal dim " + dim);
        }
        for (int i = 0; i < vec.length; i++) {
            float value = vec[i];
            if (!Float.isFinite(value) || Math.abs(value) >= MAX_INPUT_MAGNITUDE) {
                throw new IllegalArgumentException(name + "[" + i + "] must be finite and |value| < 1e16");
            }
        }
    }

    private void validatePackedCodes(byte[] packedCodes) {
        if (packedCodes == null) {
            throw new IllegalArgumentException("packedCodes must not be null");
        }
        int expectedLength = bitWidth * dim / 8;
        if (packedCodes.length != expectedLength) {
            throw new IllegalArgumentException("packedCodes length must be " + expectedLength);
        }
    }

    private void validatePositions(int[] quantizedPos) {
        if (quantizedPos == null) {
            throw new IllegalArgumentException("quantizedPos must not be null");
        }
        if (quantizedPos.length != dim) {
            throw new IllegalArgumentException("quantizedPos length must equal dim " + dim);
        }
        for (int i = 0; i < quantizedPos.length; i++) {
            if (quantizedPos[i] < 0 || quantizedPos[i] >= centroids.length) {
                throw new IllegalArgumentException("quantizedPos[" + i + "] is outside the codebook range");
            }
        }
    }

    private void validateNorm(float norm) {
        if (!Float.isFinite(norm)) {
            throw new IllegalArgumentException("norm must be finite");
        }
    }

    public int[] getCentroidPositions(float[] embedding) {
        validateVector(embedding, "embedding");
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
        validateVector(embedding, "embedding");
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

        hasQuantizedVectors = true;
        return new QuantizedVector(packedRow);
    }

    public void unpack(byte[] packedRow, int[] outCodes) {
        int bytesPerPlane = dim / 8;
        for (int j = 0; j < dim; j++) {
            int bytePos = j / 8;
            int bitPos = 7 - (j % 8);
            int code = 0;
            for (int p = 0; p < bitWidth; p++) {
                if ((packedRow[p * bytesPerPlane + bytePos] & (1 << bitPos)) != 0) {
                    code |= (1 << p);
                }
            }
            outCodes[j] = code;
        }
    }

    public int[] unpack(byte[] packedRow) {
        int[] codes = new int[dim];
        unpack(packedRow, codes);
        return codes;
    }

    public float[] convertToNormalSpace(int[] quantizedPos) {
        validatePositions(quantizedPos);
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
            result[i] = sum;
        }
        return result;
    }

    public float[] convertToNormalSpace(byte[] packedCodes) {
        int[] positions = unpack(packedCodes);
        return convertToNormalSpace(positions);
    }

    /**
     * Pre-rotates a raw query vector into the quantized space to prepare it for 
     * fast asymmetric distance calculation.
     */
    public float[] rotateQuery(float[] query) {
        float norm = computeNorm(query);
        float[] unit = new float[dim];
        if (norm > 1e-10) {
            for (int d = 0; d < dim; d++) unit[d] = query[d] / norm;
        }

        float[] rotated = new float[dim];
        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += unit[j] * rotationMatrix[d][j];
            }
            rotated[d] = sum * norm;
        }
        return rotated;
    }

    public float[][] precomputeAsymmetricLUT(float[] preRotatedQuery) {
        int numCentroids = 1 << bitWidth;
        float[][] lut = new float[dim][numCentroids];
        for (int d = 0; d < dim; d++) {
            float q = preRotatedQuery[d];
            float invS = invScaleTq[d];
            float sh = shift[d];
            for (int c = 0; c < numCentroids; c++) {
                float valDb = centroids[c] * invS - sh;
                lut[d][c] = q * valDb;
            }
        }
        return lut;
    }

    public float asymmetricDotProductLUT(float[][] lut, int[] unpackedDb) {
        float sum = 0;
        for (int d = 0; d < dim; d++) {
            sum += lut[d][unpackedDb[d]];
        }
        return sum;
    }

    public float asymmetricDotProduct(float[] preRotatedQuery, QuantizedVector dbVec) {
        int[] posDb = unpack(dbVec.getPackedCodes());
        float sum = 0;
        for (int d = 0; d < dim; d++) {
            float valDb = centroids[posDb[d]] * invScaleTq[d] - shift[d];
            sum += preRotatedQuery[d] * valDb;
        }
        return sum;
    }

    /**
     * Computes the symmetric dot product directly in the rotated space.
     * For search-style scoring, prefer
     * {@link #scoreRawQuery(float[], QuantizedVector)}.
     * This avoids the O(dim^2) reconstruction matrix multiplication.
     */
    public float symmetricDotProduct(QuantizedVector q1, QuantizedVector q2) {
        if (q1 == null || q2 == null) {
            throw new IllegalArgumentException("quantized vectors must not be null");
        }
        int[] pos1 = unpack(q1.getPackedCodes());
        int[] pos2 = unpack(q2.getPackedCodes());
        float sum = 0;
        for (int d = 0; d < dim; d++) {
            float val1 = centroids[pos1[d]] * invScaleTq[d] - shift[d];
            float val2 = centroids[pos2[d]] * invScaleTq[d] - shift[d];
            sum += val1 * val2;
        }
        return sum;
    }

    /**
     * Optimized symmetric dot product where the query is pre-unpacked into floating point rotated values.
     * Use this when scoring a single query against many database vectors.
     */
    public float symmetricDotProduct(float[] queryRotatedVals, QuantizedVector dbVec) {
        validateVector(queryRotatedVals, "queryRotatedVals");
        if (dbVec == null) {
            throw new IllegalArgumentException("dbVec must not be null");
        }
        int[] posDb = unpack(dbVec.getPackedCodes());
        float sum = 0;
        for (int d = 0; d < dim; d++) {
            float valDb = centroids[posDb[d]] * invScaleTq[d] - shift[d];
            sum += queryRotatedVals[d] * valDb;
        }
        return sum;
    }

    /**
     * Scores an unquantized query against a quantized database vector using the
     * same TQ+ inverse-query calibration math as turbovec-main search.
     */
    public float scoreRawQuery(float[] query, QuantizedVector dbVec) {
        validateVector(query, "query");
        if (dbVec == null) {
            throw new IllegalArgumentException("dbVec must not be null");
        }
        int[] posDb = unpack(dbVec.getPackedCodes());
        float[] qRot = rotate(query);
        double sum = 0.0;
        for (int d = 0; d < dim; d++) {
            double valDb = centroids[posDb[d]] * invScaleTq[d] - shift[d];
            sum += qRot[d] * valDb;
        }
        return (float) sum;
    }

    /**
     * Helper to get the rotated float values for a quantized vector directly.
     */
    public float[] getQuantizedRotatedValues(QuantizedVector q) {
        if (q == null) {
            throw new IllegalArgumentException("q must not be null");
        }
        int[] pos = unpack(q.getPackedCodes());
        float[] vals = new float[dim];
        for (int d = 0; d < dim; d++) {
            vals[d] = centroids[pos[d]] * invScaleTq[d] - shift[d];
        }
        return vals;
    }

    private float[] rotate(float[] vector) {
        float[] rotated = new float[dim];
        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += vector[j] * rotationMatrix[d][j];
            }
            rotated[d] = sum;
        }
        return rotated;
    }
}
