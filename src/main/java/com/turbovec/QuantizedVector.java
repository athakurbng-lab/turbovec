package com.turbovec;

import java.util.Arrays;

public class QuantizedVector {
    private final byte[] packedCodes;
    private final byte[] qjlBits;
    private final float qjlScale;

    public QuantizedVector(byte[] packedCodes, byte[] qjlBits, float qjlScale) {
        if (packedCodes == null) {
            throw new IllegalArgumentException("packedCodes must not be null");
        }
        this.packedCodes = Arrays.copyOf(packedCodes, packedCodes.length);
        this.qjlBits = qjlBits != null ? Arrays.copyOf(qjlBits, qjlBits.length) : null;
        this.qjlScale = qjlScale;
    }

    public QuantizedVector(byte[] packedCodes) {
        this(packedCodes, null, 0f);
    }

    public byte[] getPackedCodes() {
        return Arrays.copyOf(packedCodes, packedCodes.length);
    }

    public byte[] getQjlBits() {
        return qjlBits != null ? Arrays.copyOf(qjlBits, qjlBits.length) : null;
    }

    public float getQjlScale() {
        return qjlScale;
    }
}
