package com.turbovec;

import java.util.Arrays;

public class QuantizedVector {
    private final byte[] packedCodes;
    private final float norm;

    public QuantizedVector(byte[] packedCodes, float norm) {
        if (packedCodes == null) {
            throw new IllegalArgumentException("packedCodes must not be null");
        }
        if (!Float.isFinite(norm)) {
            throw new IllegalArgumentException("norm must be finite");
        }
        this.packedCodes = Arrays.copyOf(packedCodes, packedCodes.length);
        this.norm = norm;
    }

    public byte[] getPackedCodes() {
        return Arrays.copyOf(packedCodes, packedCodes.length);
    }

    public float getNorm() {
        return norm;
    }
}
