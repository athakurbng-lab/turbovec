# turbovec-java

A complete, high-performance Java implementation of the TurboQuant algorithm logic originally developed in [turbovec](https://github.com/RyanCodrai/turbovec).

## Features

- **Blazing Fast ADC Scoring:** Implements Asymmetric Dot Product (ADC) using precomputed Lookup Tables (LUTs) and zero-allocation unpacking to score millions of vectors per second without heavy float multiplications.
- **Quantization:** Converts float embeddings into highly compressed codes (4, 8, or 16 bits) using random orthogonal rotations and Beta-distribution Lloyd-Max codebooks.
- **TQ+ Calibration:** Mathematically aligns vector margins by shifting and scaling empirically observed quantiles onto standard distributions.
- **Automated Experimentation:** A dedicated benchmark suite that analyzes the recall tradeoff across bit-widths, codebooks, calibration sizes, and rotation strategies.

## Usage

```java
int dim = 384; 
int bitWidth = 8;
long seed = 100L;
boolean fillRowFirst = false;

TurboVec turbovec = new TurboVec(dim, bitWidth, seed, fillRowFirst);

// Fit TQ+ Calibration on a sample of your database (Mandatory for high accuracy)
turbovec.fitCalibration(databaseSample);

// Quantize vectors to bit-packed byte arrays
QuantizedVector qVec = turbovec.quantize(embedding);

// --- Fast Asymmetric Search (ADC) ---
// 1. Rotate the high-precision query once
float[] rotatedQuery = turbovec.rotateQuery(query);

// 2. Precompute a Lookup Table (LUT) for instant scoring
float[][] lut = turbovec.precomputeAsymmetricLUT(rotatedQuery);

// 3. Score against the database instantly using the LUT
int[] unpackedDB = turbovec.unpack(qVec.getPackedCodes());
float score = turbovec.asymmetricDotProductLUT(lut, unpackedDB);
```

## Running the Automated Benchmark

The project includes `ExperimentBenchmark.java`, which automates testing and outputs a formatted table comparing the exact intersection (Recall) of the highest-performing configurations across 4-bit, 8-bit, and 16-bit architectures.

### Step 1: Prepare Your Data (Optional)
To test against real data, drop two text files into the project root:
- `database.txt`
- `queries.txt`

**Format requirements:**
- One vector per line.
- Values must be floats separated by spaces or commas (e.g., `0.012, -0.34, 0.99...`).
- The benchmark expects the files to match the hardcoded `dim=384` dimension size (update the variable inside `ExperimentBenchmark.java` if using a different dimension).
- *Fallback:* If these files are not found, the benchmark gracefully defaults to generating 50,000 random unit vectors for the database and 1,000 queries.

### Step 2: Execute the Script
Run the following Maven command from your terminal:
```bash
mvn test-compile exec:java -Dexec.classpathScope="test" -Dexec.mainClass="com.turbovec.ExperimentBenchmark"
```

### Step 3: Understanding the Report
The benchmark normalizes all loaded vectors to unit magnitude, calculates the **exact** ground truth by performing full precision dot-products, and then executes the quantized evaluation.

The final report evaluates three specific optimized configurations:
1. **4-bit** (`25%` calibration size, `seed=0`, `rowFirst=true`) - The highest optimized setup for 4-bit Asymmetric Search.
2. **8-bit** (`50%` calibration size, `seed=100`, `rowFirst=false`) - Pushes recall higher while maintaining memory efficiency via 8-bit quantization.
3. **16-bit** (`10%` calibration size, `seed=0`, `rowFirst=false`) - Extreme precision configuration using a 65,536-level codebook.

The script evaluates the results against the exact ground truth to report **Recall@5**, **Recall@10**, and **Recall@15** (calculated via exact array intersection).
