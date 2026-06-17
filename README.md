# turbovec-java

A complete, dependency-free (except for `commons-math3` for Beta distribution and QR decomposition) Java wrapper for the TurboQuant algorithm logic from [turbovec](https://github.com/RyanCodrai/turbovec).

## Features

- **Quantization:** Converts float embeddings into quantized codes using a deterministic random orthogonal rotation matrix, Beta distribution Lloyd-Max codebooks, and TQ+ calibration.
- **Reconstruction:** Converts quantized bit-packed codes back to approximated vectors in normal space.
- **Similarity Search Evaluation:** Includes a recall benchmark out-of-the-box (`RecallBenchmark.java`).

## Usage

```java
int dim = 1536; // OpenAI dim
int bitWidth = 4; // 4-bit precision
TurboVec turbovec = new TurboVec(dim, bitWidth);

// Fit TQ+ Calibration (optional but highly recommended for accuracy)
// turbovec.fitCalibration(batchOfEmbeddings);

// Quantize to bit-packed array
QuantizedVector qVec = turbovec.quantize(embedding);

// Reconstruct
float[] normalSpaceVec = turbovec.convertToNormalSpace(qVec.getPackedCodes(), qVec.getNorm());
```

## Running Benchmarks

```bash
mvn test-compile exec:java -Dexec.classpathScope="test" -Dexec.mainClass="com.turbovec.RecallBenchmark"
```
