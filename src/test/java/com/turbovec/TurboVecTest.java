package com.turbovec;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TurboVecTest {
    private static final float EPS = 1e-6f;

    @Test
    public void constructorRejectsUnsupportedParameters() {
        assertThrows(IllegalArgumentException.class, () -> new TurboVec(8, 1));
        assertThrows(IllegalArgumentException.class, () -> new TurboVec(8, 5));
        assertThrows(IllegalArgumentException.class, () -> new TurboVec(0, 4));
        assertThrows(IllegalArgumentException.class, () -> new TurboVec(10, 4));
        assertThrows(IllegalArgumentException.class, () -> new TurboVec(TurboVec.MAX_DIM + 8, 4));
    }

    @Test
    public void quantizedVectorDefensivelyCopiesPackedCodes() {
        byte[] packed = new byte[]{1, 2, 3};
        QuantizedVector vector = new QuantizedVector(packed, 1.0f);

        packed[0] = 99;
        assertEquals(1, vector.getPackedCodes()[0]);

        byte[] fromGetter = vector.getPackedCodes();
        fromGetter[1] = 88;
        assertEquals(2, vector.getPackedCodes()[1]);
    }

    @Test
    public void rejectsInvalidVectorsAndPackedCodes() {
        TurboVec turboVec = new TurboVec(8, 2);
        assertThrows(IllegalArgumentException.class, () -> turboVec.quantize(null));
        assertThrows(IllegalArgumentException.class, () -> turboVec.quantize(new float[7]));

        float[] withNaN = new float[8];
        withNaN[3] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> turboVec.quantize(withNaN));

        float[] withInfinity = new float[8];
        withInfinity[4] = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> turboVec.getCentroidPositions(withInfinity));

        float[] huge = new float[8];
        huge[2] = 1e16f;
        assertThrows(IllegalArgumentException.class, () -> turboVec.scoreRawQuery(huge, new QuantizedVector(new byte[2], 1.0f)));

        assertThrows(IllegalArgumentException.class, () -> turboVec.unpack(new byte[1]));
        assertThrows(IllegalArgumentException.class, () -> turboVec.convertToNormalSpace(new int[]{0, 1}, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> turboVec.convertToNormalSpace(new int[8], Float.NaN));
    }

    @Test
    public void rejectsInvalidCalibrationBatchesAndRefitAfterQuantization() {
        TurboVec turboVec = new TurboVec(8, 4);
        assertThrows(IllegalArgumentException.class, () -> turboVec.fitCalibration(null));
        assertThrows(IllegalArgumentException.class, () -> turboVec.fitCalibration(new float[][]{null}));
        assertThrows(IllegalArgumentException.class, () -> turboVec.fitCalibration(new float[][]{new float[7]}));

        turboVec.fitCalibration(new float[][]{new float[8]});
        turboVec.quantize(new float[8]);
        assertThrows(IllegalStateException.class, () -> turboVec.fitCalibration(new float[][]{new float[8]}));
    }

    @Test
    public void codebookMatchesKnownMainProjectValuesFor1536Dimensions() {
        assertCodebook(
                2,
                new float[]{-0.025040101f, -9.5513875E-15f, 0.025040101f},
                new float[]{-0.038527973f, -0.0115522295f, 0.0115522295f, 0.038527973f}
        );
        assertCodebook(
                3,
                new float[]{-0.044569556f, -0.02677956f, -0.0127682295f, -2.1243424E-14f, 0.0127682295f, 0.02677956f, 0.044569556f},
                new float[]{-0.054864164f, -0.034274943f, -0.019284176f, -0.006252284f, 0.006252284f, 0.019284176f, 0.034274943f, 0.054864164f}
        );
        assertCodebook(
                4,
                new float[]{-0.061185353f, -0.046998676f, -0.036645103f, -0.028033737f, -0.020391596f, -0.013324004f, -0.0065861857f, -3.1358813E-14f, 0.0065861857f, 0.013324004f, 0.020391596f, 0.028033737f, 0.036645103f, 0.046998676f, 0.061185353f},
                new float[]{-0.06962875f, -0.05274195f, -0.041255407f, -0.0320348f, -0.024032677f, -0.016750518f, -0.009897489f, -0.003274882f, 0.003274882f, 0.009897489f, 0.016750518f, 0.024032677f, 0.0320348f, 0.041255407f, 0.05274195f, 0.06962875f}
        );
    }

    @Test
    public void quantizationRoundTripsForSupportedBitWidths() {
        for (int bits = 2; bits <= 4; bits++) {
            TurboVec turboVec = new TurboVec(64, bits);
            float[] vector = randomVector(64, 1234 + bits);

            QuantizedVector quantized = turboVec.quantize(vector);
            assertEquals(64 / 8 * bits, quantized.getPackedCodes().length);
            assertEquals(64, turboVec.unpack(quantized.getPackedCodes()).length);

            float[] reconstructed = turboVec.convertToNormalSpace(quantized.getPackedCodes(), quantized.getNorm());
            assertEquals(64, reconstructed.length);
            for (float value : reconstructed) {
                assertTrue(Float.isFinite(value));
            }
        }
    }

    @Test
    public void zeroVectorReconstructsToZero() {
        TurboVec turboVec = new TurboVec(64, 4);
        QuantizedVector quantized = turboVec.quantize(new float[64]);
        assertEquals(0.0f, quantized.getNorm(), 0.0f);

        float[] reconstructed = turboVec.convertToNormalSpace(quantized.getPackedCodes(), quantized.getNorm());
        for (float value : reconstructed) {
            assertEquals(0.0f, value, 0.0f);
        }
    }

    @Test
    public void rawQueryScoringMatchesReconstructionDotProduct() {
        TurboVec turboVec = new TurboVec(64, 4);
        float[] databaseVector = randomVector(64, 7);
        float[] query = randomVector(64, 8);

        QuantizedVector quantized = turboVec.quantize(databaseVector);
        float score = turboVec.scoreRawQuery(query, quantized);
        float[] reconstructed = turboVec.convertToNormalSpace(quantized.getPackedCodes(), quantized.getNorm());

        assertEquals(dot(query, reconstructed), score, 1e-4f);
    }

    private static void assertCodebook(int bits, float[] expectedBoundaries, float[] expectedCentroids) {
        float[][] codebook = TurboVec.generateLloydMaxCodebook(bits, 1536);
        assertArrayEquals(expectedBoundaries, codebook[0], EPS);
        assertArrayEquals(expectedCentroids, codebook[1], EPS);
    }

    private static float[] randomVector(int dim, long seed) {
        Random random = new Random(seed);
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = (float) random.nextGaussian();
        }
        return vector;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
