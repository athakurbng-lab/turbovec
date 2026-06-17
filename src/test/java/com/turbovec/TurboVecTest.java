package com.turbovec;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Random;

public class TurboVecTest {

    @Test
    public void testTurboVecEndToEnd() {
        int dim = 1536; // OpenAI dim
        int bits = 4;
        TurboVec turbovec = new TurboVec(dim, bits);
        
        // Ensure centroid array is correct size
        float[] centroids = turbovec.getCentroidArray();
        assertEquals(16, centroids.length); // 2^4 = 16

        // Fit calibration on a random batch
        int batchSize = 1000;
        float[][] batch = new float[batchSize][dim];
        Random rand = new Random(123);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < dim; j++) {
                batch[i][j] = (float) rand.nextGaussian();
            }
        }
        turbovec.fitCalibration(batch);

        // Quantize a vector
        float[] testEmbedding = new float[dim];
        for (int j = 0; j < dim; j++) {
            testEmbedding[j] = (float) rand.nextGaussian();
        }

        QuantizedVector qVec = turbovec.quantize(testEmbedding);
        assertNotNull(qVec.getPackedCodes());
        assertEquals(dim / 8 * bits, qVec.getPackedCodes().length);
        
        // Unpack positions
        int[] positions = turbovec.unpack(qVec.getPackedCodes());
        assertEquals(dim, positions.length);
        
        // Also check getCentroidPositions matches exactly
        int[] directPositions = turbovec.getCentroidPositions(testEmbedding);
        for (int i = 0; i < dim; i++) {
            assertEquals(directPositions[i], positions[i], "Mismatch at index " + i);
        }

        // Convert back to normal space
        float[] reconstructed = turbovec.convertToNormalSpace(qVec.getPackedCodes(), qVec.getNorm());
        assertEquals(dim, reconstructed.length);

        // Check cosine similarity
        float dot = 0;
        float norm1 = 0;
        float norm2 = 0;
        for (int i = 0; i < dim; i++) {
            dot += testEmbedding[i] * reconstructed[i];
            norm1 += testEmbedding[i] * testEmbedding[i];
            norm2 += reconstructed[i] * reconstructed[i];
        }
        double cosineSim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        
        // Assert high cosine similarity, typically > 0.95 for 4-bit 1536d
        assertTrue(cosineSim > 0.90, "Cosine similarity too low: " + cosineSim);
    }
}
