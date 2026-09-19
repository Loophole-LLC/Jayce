// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce.intervention;

import company.loophole.jayce.ImageSamples;
import company.loophole.jayce.MnistLoader;
import company.loophole.jayce.ShapePatterns;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/** Builds the same repeatable image and shape examples for both learners. */
final class HardBenchmarkData {
    static final int[] PHASE_ORDER = {0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 3, 1};
    static final int BATCH_SIZE = 16;
    static final int BATCHES_PER_PHASE = 64;
    static final double[] ROTATION_DEGREES = {0, 15, -15, 30};
    static final String[] DATASETS = {
            "shapes-permutations", "shapes-mixed", "mnist-rotations", "fashion-rotations"
    };
    static final String[] BUDGETS = {"examples", "work", "memory"};

    record Data(double[][] x, double[][] y) {}

    /** Keeps the full image test set and a smaller balanced set for choosing settings. */
    static final class ImageEvaluation {
        final Data[] full = new Data[4];
        final Data[] checkpoints = new Data[4];

        ImageEvaluation(Data source, int examplesPerClass) {
            int[] chosen = stratifiedIndices(source, examplesPerClass);
            for (int condition = 0; condition < 4; condition++) {
                full[condition] = rotate(source, condition);
                double[][] x = new double[chosen.length][];
                double[][] y = new double[chosen.length][];
                for (int i = 0; i < chosen.length; i++) {
                    x[i] = full[condition].x[chosen[i]];
                    y[i] = full[condition].y[chosen[i]];
                }
                checkpoints[condition] = new Data(x, y);
            }
        }
    }

    /** Makes repeatable shape examples whose pixel patterns change between phases. */
    static final class ShapeStream {
        final String dataset;
        final long seed;
        final int dimensions = 64;
        final int classes = 4;
        final long phaseWork = 2_500_000;
        final Data[] evaluation;
        private final int[][] permutations = new int[4][dimensions];

        ShapeStream(String dataset, long seed) {
            if (!dataset.equals("shapes-permutations") && !dataset.equals("shapes-mixed")) {
                throw new IllegalArgumentException("Unknown shape stream: " + dataset);
            }
            this.dataset = dataset;
            this.seed = seed;
            for (int condition = 0; condition < 4; condition++) {
                for (int pixel = 0; pixel < dimensions; pixel++) permutations[condition][pixel] = pixel;
                if (condition > 0) shuffle(permutations[condition], new Random(20270918L + 7919L * condition));
            }
            Data base = shapes(new Random(seed + 90_000_000L), 200);
            evaluation = new Data[4];
            for (int condition = 0; condition < 4; condition++) evaluation[condition] = transform(base, condition);
        }

        Data batch(int phase, int batchIndex) {
            checkStreamPosition(phase, batchIndex);
            Random random = new Random(seed + 10_000_019L * phase + 1009L * batchIndex);
            Data base = shapes(random, BATCH_SIZE);
            return transform(base, PHASE_ORDER[phase]);
        }

        private Data transform(Data base, int condition) {
            double[][] x = new double[base.x.length][];
            for (int example = 0; example < x.length; example++) {
                x[example] = new double[dimensions];
                for (int pixel = 0; pixel < dimensions; pixel++) {
                    int source = dataset.equals("shapes-mixed")
                            ? (condition < 2 ? pixel : permutations[1][pixel])
                            : permutations[condition][pixel];
                    double value = base.x[example][source];
                    if (dataset.equals("shapes-mixed") && condition % 2 == 1) value = -value;
                    x[example][pixel] = value;
                }
            }
            return new Data(x, base.y);
        }
    }

    /** Makes a repeatable image batch for one point in the phase schedule. */
    static final class ImageStream {
        final long seed;
        final long phaseWork = 16_000_000;
        final Data train;
        final ImageEvaluation evaluation;

        ImageStream(long seed, Data train, ImageEvaluation evaluation) {
            this.seed = seed;
            this.train = train;
            this.evaluation = evaluation;
        }

        Data batch(int phase, int batchIndex) {
            checkStreamPosition(phase, batchIndex);
            Random random = new Random(seed + 10_000_019L * phase + 1009L * batchIndex);
            return rotate(sample(train, random, BATCH_SIZE), PHASE_ORDER[phase]);
        }
    }

    private HardBenchmarkData() {}

    static Data shapes(Random random, int count) {
        double[][] x = new double[count][];
        double[][] y = new double[count][4];
        for (int i = 0; i < count; i++) {
            int label = i % 4;
            x[i] = ImageSamples.toPixelVector(ShapePatterns.draw(label, 8, random), 8, 8);
            for (int pixel = 0; pixel < 64; pixel++) {
                if (random.nextDouble() < .08) x[i][pixel] = 1 - x[i][pixel];
                x[i][pixel] = 2 * x[i][pixel] - 1;
            }
            y[i][label] = 1;
        }
        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        shuffle(order, random);
        double[][] shuffledX = new double[count][];
        double[][] shuffledY = new double[count][];
        for (int i = 0; i < count; i++) {
            shuffledX[i] = x[order[i]];
            shuffledY[i] = y[order[i]];
        }
        return new Data(shuffledX, shuffledY);
    }

    static Data sample(Data source, Random random, int count) {
        double[][] x = new double[count][];
        double[][] y = new double[count][];
        for (int i = 0; i < count; i++) {
            int selected = random.nextInt(source.x.length);
            x[i] = source.x[selected];
            y[i] = source.y[selected];
        }
        return new Data(x, y);
    }

    static Data reduce(MnistLoader.Dataset source, int first, int last) {
        if (source.imageWidth() != 28 || source.imageHeight() != 28) {
            throw new IllegalArgumentException("Expected 28 by 28 MNIST images");
        }
        double[][] x = new double[last - first][196];
        double[][] y = new double[last - first][];
        for (int image = first; image < last; image++) {
            y[image - first] = source.targets()[image].clone();
            for (int row = 0; row < 14; row++) {
                for (int column = 0; column < 14; column++) {
                    int offset = 56 * row + 2 * column;
                    double[] pixels = source.inputs()[image];
                    x[image - first][14 * row + column] =
                            (pixels[offset] + pixels[offset + 1] + pixels[offset + 28] + pixels[offset + 29]) / 2 - 1;
                }
            }
        }
        return new Data(x, y);
    }

    private static Data rotate(Data source, int condition) {
        double[][] x = new double[source.x.length][];
        for (int i = 0; i < x.length; i++) x[i] = rotate(source.x[i], ROTATION_DEGREES[condition]);
        return new Data(x, source.y);
    }

    private static double[] rotate(double[] input, double degrees) {
        double[] output = new double[196];
        double angle = Math.toRadians(degrees);
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);
        for (int row = 0; row < 14; row++) {
            for (int column = 0; column < 14; column++) {
                double sourceX = cosine * (column - 6.5) + sine * (row - 6.5) + 6.5;
                double sourceY = -sine * (column - 6.5) + cosine * (row - 6.5) + 6.5;
                int left = (int) Math.floor(sourceX);
                int top = (int) Math.floor(sourceY);
                double fractionX = sourceX - left;
                double fractionY = sourceY - top;
                output[14 * row + column] = (1 - fractionY) * (
                        (1 - fractionX) * pixel(input, left, top)
                                + fractionX * pixel(input, left + 1, top))
                        + fractionY * ((1 - fractionX) * pixel(input, left, top + 1)
                                + fractionX * pixel(input, left + 1, top + 1));
            }
        }
        return output;
    }

    private static double pixel(double[] input, int column, int row) {
        if (column < 0 || column >= 14 || row < 0 || row >= 14) return -1;
        return input[row * 14 + column];
    }

    private static int[] stratifiedIndices(Data source, int examplesPerClass) {
        if (examplesPerClass < 1) throw new IllegalArgumentException("Positive subset size required");
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < 10; c++) byClass.add(new ArrayList<>());
        for (int i = 0; i < source.y.length; i++) byClass.get(argmax(source.y[i])).add(i);
        int[] selected = new int[10 * examplesPerClass];
        Random random = new Random(20260918L);
        for (int c = 0; c < 10; c++) {
            if (byClass.get(c).size() < examplesPerClass) {
                throw new IllegalArgumentException("Not enough validation examples for every class");
            }
            Collections.shuffle(byClass.get(c), random);
            for (int i = 0; i < examplesPerClass; i++) selected[c * examplesPerClass + i] = byClass.get(c).get(i);
        }
        shuffle(selected, random);
        return selected;
    }

    static int argmax(double[] values) {
        int best = 0;
        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) throw new IllegalStateException("Non-finite model output");
            if (values[i] > values[best]) best = i;
        }
        return best;
    }

    private static void checkStreamPosition(int phase, int batch) {
        if (phase < 0 || phase >= PHASE_ORDER.length || batch < 0 || batch >= 4096) {
            throw new IllegalArgumentException("Training stream position is out of bounds");
        }
    }

    private static void shuffle(int[] values, Random random) {
        for (int i = values.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temporary = values[i];
            values[i] = values[j];
            values[j] = temporary;
        }
    }
}
