// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce.apm;

import java.util.Arrays;

/**
 * A beginner-readable Java example of Adaptive Prototype Memory (APM), the
 * learning method used by JAYCE (Jayce Associates Your Categorized Exemplars).
 *
 * <p>An exemplar is a specific example kept for comparison. Each prototype starts
 * as a copy of one labeled input: a list of numbers. When a labeled example
 * arrives, APM saves it in an empty slot for that label. If the slots are full,
 * APM moves the closest prototype for that label partway toward the new input.
 * An updated prototype can therefore blend information from several examples.
 * To guess a label, APM finds the closest prototype across all labels.
 *
 * <p>Think of each label as a toy bin. The project's toy story was inspired
 * by toddler Jayce. The person using this class supplies the label; the code
 * does not recognize toys or decide whether its own guess was correct.
 *
 * <p>Inputs must contain finite numbers from -1 to 1, and labels are
 * numbered from 0 to {@code classes - 1}.
 */
public final class JayceMemory {
    /**
     * Counts what the learner has done so far.
     *
     * @param examples number of individual examples learned
     * @param updates number of calls that changed memory
     * @param distanceComponents number of input values checked while comparing examples
     * @param movedComponents number of prototype values changed
     * @param seededComponents number of values copied into new prototype slots
     * @param work rough operation count used by the benchmark
     * @param activePrototypes number of filled memory slots
     */
    public record Statistics(long examples, long updates, long distanceComponents,
                             long movedComponents, long seededComponents, long work,
                             int activePrototypes) {}

    // Number of values in one input vector.
    private final int dimensions;
    // Number of labels the model can predict.
    private final int classes;
    // How many saved vectors each label is allowed to keep.
    private final int prototypesPerClass;
    // Controls how sharply scores favor the closest label.
    private final double temperature;
    // prototypes[label][slot][number]: saved vectors grouped by label.
    private final double[][][] prototypes;
    // Number of filled memory slots for each label.
    private final int[] active;
    private long examples;
    private long updates;
    private long distanceComponents;
    private long movedComponents;
    private long seededComponents;
    private long work;

    /**
     * Creates an empty memory with the same number of slots for every label.
     *
     * @param dimensions number of values in each input
     * @param classes number of labels, numbered from zero
     * @param prototypesPerClass maximum prototypes for each label
     * @param temperature controls how sharply {@link #scores(double[])} favors closer prototypes
     */
    public JayceMemory(int dimensions, int classes, int prototypesPerClass,
                                   double temperature) {
        if (dimensions < 1 || classes < 2 || prototypesPerClass < 1
                || !Double.isFinite(temperature) || temperature < 1e-6) {
            throw new IllegalArgumentException(
                    "Positive dimensions, at least two classes, prototypes, and temperature are required");
        }
        this.dimensions = dimensions;
        this.classes = classes;
        this.prototypesPerClass = prototypesPerClass;
        this.temperature = temperature;
        // Slots start empty. The last array holds one saved input vector.
        this.prototypes = new double[classes][prototypesPerClass][];
        this.active = new int[classes];
    }

    /**
     * Learns several labeled examples in order.
     *
     * @param inputs the number lists to learn
     * @param labels the correct label for each input
     * @param rate how far a full group's closest prototype moves toward an input, from 0 to 1
     */
    public void train(double[][] inputs, int[] labels, double rate) {
        validateBatch(inputs, labels, rate);
        for (int n = 0; n < inputs.length; n++) updateOne(inputs[n], labels[n], rate);
        examples += inputs.length;
        updates++;
    }

    /**
     * Learns one labeled example.
     *
     * @param input the number list to learn
     * @param label the correct label for this input
     * @param rate how far a full group's closest prototype moves toward the input, from 0 to 1
     */
    public void train(double[] input, int label, double rate) {
        validateInput(input);
        validateLabel(label);
        validateRate(rate);
        updateOne(input, label, rate);
        examples++;
        updates++;
    }

    /**
     * Guesses by returning the label of the closest prototype.
     *
     * @param input the number list to classify
     * @return the closest prototype's label, or -1 if memory is empty
     */
    public int predictClass(double[] input) {
        validateInput(input);
        int bestClass = -1;
        double minimum = Double.POSITIVE_INFINITY;
        // Check each saved vector and keep the label of the closest one.
        for (int clazz = 0; clazz < classes; clazz++) {
            for (int slot = 0; slot < active[clazz]; slot++) {
                double distance = distance(input, prototypes[clazz][slot]);
                if (distance < minimum) {
                    minimum = distance;
                    bestClass = clazz;
                }
            }
        }
        return bestClass;
    }

    /**
     * Returns one closeness score per label. The scores add to 1, but are not
     * a guaranteed estimate of the chance that each label is correct.
     *
     * @param input the number list to classify
     * @return scores in label order, from label 0 through the last label
     */
    public double[] scores(double[] input) {
        validateInput(input);
        // One distance per label: distance to that label's closest saved vector.
        double[] distances = new double[classes];
        Arrays.fill(distances, Double.POSITIVE_INFINITY);
        double minimum = Double.POSITIVE_INFINITY;
        for (int clazz = 0; clazz < classes; clazz++) {
            for (int slot = 0; slot < active[clazz]; slot++) {
                distances[clazz] = Math.min(distances[clazz],
                        distance(input, prototypes[clazz][slot]) / dimensions);
            }
            minimum = Math.min(minimum, distances[clazz]);
        }
        double[] result = new double[classes];
        if (!Double.isFinite(minimum)) {
            Arrays.fill(result, 1.0 / classes);
            return result;
        }
        // Convert distances into positive scores; closer labels get larger scores.
        double total = 0;
        for (int clazz = 0; clazz < classes; clazz++) {
            result[clazz] = Math.exp(-(distances[clazz] - minimum) / temperature);
            total += result[clazz];
        }
        // Scale the scores so they add up to 1.
        for (int clazz = 0; clazz < classes; clazz++) result[clazz] /= total;
        return result;
    }

    /**
     * Returns a copy of the number of active prototypes for each label.
     *
     * @return one count for each label
     */
    public int[] activePrototypesPerClass() { return active.clone(); }

    /**
     * Estimates a safe upper limit on the work for a batch, for the benchmark.
     *
     * @param examplesInBatch number of examples in the batch
     * @return an upper-bound estimate, not a timing measurement
     */
    public long batchWorkUpperBound(int examplesInBatch) {
        if (examplesInBatch < 1) throw new IllegalArgumentException("Batch must not be empty");
        int emptySlots = classes * prototypesPerClass - Arrays.stream(active).sum();
        long possibleSeeds = Math.min(examplesInBatch, emptySlots);
        long maximumSearchAndMove = prototypesPerClass * (3L * dimensions + 1) + 3L * dimensions;
        return possibleSeeds * dimensions + 2L * examplesInBatch
                + examplesInBatch * maximumSearchAndMove;
    }

    /**
     * Returns counts of the examples and prototype operations so far.
     *
     * @return the current statistics
     */
    public Statistics statistics() {
        return new Statistics(examples, updates, distanceComponents, movedComponents,
                seededComponents, work, Arrays.stream(active).sum());
    }

    /**
     * Estimates bytes used by filled prototypes and per-label counters.
     * This leaves out Java object overhead.
     *
     * @return estimated bytes currently storing the model
     */
    public long persistentBytes() {
        return 8L * dimensions * Arrays.stream(active).sum() + 4L * classes;
    }

    /**
     * Estimates bytes needed if every prototype slot is filled.
     * This leaves out Java object overhead.
     *
     * @return estimated bytes for the model's full memory capacity
     */
    public long capacityBytes() {
        return 8L * dimensions * classes * prototypesPerClass + 4L * classes;
    }

    // Add one input to its labeled group, or nudge that group's closest vector.
    private void updateOne(double[] input, int clazz, double rate) {
        if (active[clazz] < prototypesPerClass) {
            // Fill empty slots first. Keep a copy so outside code cannot change memory.
            prototypes[clazz][active[clazz]++] = input.clone();
            seededComponents += dimensions;
            work += dimensions;
            return;
        }
        // All slots are full: find the closest saved vector for this label.
        int nearest = nearestWithin(input, clazz);
        double[] prototype = prototypes[clazz][nearest];
        // Move each number partway toward the new input. A rate of 0.1 moves
        // each value one tenth of the distance across the gap.
        for (int d = 0; d < dimensions; d++) {
            prototype[d] += rate * (input[d] - prototype[d]);
        }
        distanceComponents += (long) active[clazz] * dimensions;
        movedComponents += dimensions;
        work += active[clazz] * (3L * dimensions + 1) + 3L * dimensions;
    }

    // Find the closest vector among one label's saved vectors.
    private int nearestWithin(double[] input, int clazz) {
        int best = 0;
        double minimum = Double.POSITIVE_INFINITY;
        for (int slot = 0; slot < active[clazz]; slot++) {
            double distance = distance(input, prototypes[clazz][slot]);
            if (distance < minimum) {
                minimum = distance;
                best = slot;
            }
        }
        return best;
    }

    // Add up squared differences: a smaller sum means the vectors are closer.
    private double distance(double[] input, double[] prototype) {
        double result = 0;
        for (int d = 0; d < dimensions; d++) {
            double delta = input[d] - prototype[d];
            result += delta * delta;
        }
        return result;
    }

    // Check the whole batch before changing memory, so bad data fails early.
    private void validateBatch(double[][] inputs, int[] labels, double rate) {
        if (inputs == null || labels == null || inputs.length == 0
                || inputs.length != labels.length) {
            throw new IllegalArgumentException("Nonempty paired batch required");
        }
        validateRate(rate);
        for (int n = 0; n < inputs.length; n++) {
            validateInput(inputs[n]);
            validateLabel(labels[n]);
        }
    }

    // Labels must be numbered from 0 up to classes - 1.
    private void validateLabel(int label) {
        if (label < 0 || label >= classes) throw new IllegalArgumentException("Label out of range");
    }

    // A learning rate is a fraction: greater than 0 and at most 1.
    private void validateRate(double rate) {
        if (!Double.isFinite(rate) || rate <= 0 || rate > 1) {
            throw new IllegalArgumentException("Rate in (0,1] required");
        }
    }

    // Inputs must have the right number of finite values in the agreed range.
    private void validateInput(double[] input) {
        if (input == null || input.length != dimensions) throw new IllegalArgumentException("Wrong input dimension");
        for (double value : input) {
            if (!Double.isFinite(value) || value < -1 || value > 1) {
                throw new IllegalArgumentException("Inputs must be finite and in [-1,1]");
            }
        }
    }
}
