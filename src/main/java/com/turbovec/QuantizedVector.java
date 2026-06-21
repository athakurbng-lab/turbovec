package com.turbovec;

import java.util.Arrays;

public class QuantizedVector {
    private final byte[] packedCodes;

    public QuantizedVector(byte[] packedCodes) {
        if (packedCodes == null) {
            throw new IllegalArgumentException("packedCodes must not be null");
        }
        this.packedCodes = Arrays.copyOf(packedCodes, packedCodes.length);
    }

    public byte[] getPackedCodes() {
        return Arrays.copyOf(packedCodes, packedCodes.length);
    }
}
