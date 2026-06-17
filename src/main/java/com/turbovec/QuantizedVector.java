package com.turbovec;

public class QuantizedVector {
    private final byte[] packedCodes;
    private final float norm;

    public QuantizedVector(byte[] packedCodes, float norm) {
        this.packedCodes = packedCodes;
        this.norm = norm;
    }

    public byte[] getPackedCodes() {
        return packedCodes;
    }

    public float getNorm() {
        return norm;
    }
}
