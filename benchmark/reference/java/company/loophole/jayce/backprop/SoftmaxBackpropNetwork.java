// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce.backprop;

import java.util.Arrays;
import java.util.Random;

/**
 * A small, conventional neural network trained by backpropagation.
 *
 * <p>The network has an input layer, one hidden layer, and one output for each
 * label. ReLU keeps positive hidden values and changes negative ones to zero.
 * Softmax turns the outputs into scores that add to 1. The training code
 * compares those scores with the correct answer, then uses the chain rule to
 * work backward and find how each weight affected the error. Adam uses running
 * averages of those changes to adjust the weights.
 *
 * <p>This is the backpropagation model used in the JAYCE benchmark. It learns
 * only from the examples it is given; it has no replay buffer or machine-learning
 * library.
 */
public final class SoftmaxBackpropNetwork {
    private static final double ADAM_BETA_1 = 0.9;
    private static final double ADAM_BETA_2 = 0.999;
    private static final double ADAM_EPSILON = 1e-8;

    // Sizes of the input, hidden layer, and output classes.
    private final int inputCount;
    private final int hiddenCount;
    private final int classCount;
    // First index in weights where the output-layer values begin.
    private final int outputLayerOffset;

    // Weights and biases are stored together in one flat array.
    private final double[] weights;
    // Adam remembers an average gradient and an average squared gradient per weight.
    private final double[] firstMoment;
    private final double[] secondMoment;
    private final Random random;
    private long updateCount;

    /**
     * Creates a network and gives its weights small, repeatable starting values.
     *
     * @param inputs number of values in each input
     * @param hidden number of neurons in the hidden layer
     * @param classes number of labels the network can predict
     * @param seed starting value used to make the random weights repeatable
     */
    public SoftmaxBackpropNetwork(int inputs, int hidden, int classes, long seed) {
        if (inputs < 1 || hidden < 1 || classes < 2) {
            throw new IllegalArgumentException("Invalid network dimensions");
        }
        inputCount = inputs;
        hiddenCount = hidden;
        classCount = classes;
        outputLayerOffset = hiddenCount * (inputCount + 1);
        weights = new double[outputLayerOffset + classCount * (hiddenCount + 1)];
        firstMoment = new double[weights.length];
        secondMoment = new double[weights.length];
        random = new Random(seed);

        // Start with small random weights so different neurons can learn different patterns.
        for (int neuron = 0; neuron < hiddenCount; neuron++) {
            for (int input = 0; input < inputCount; input++) {
                weights[neuron * (inputCount + 1) + input]
                        = random.nextGaussian() * Math.sqrt(2.0 / inputCount);
            }
        }
        for (int clazz = 0; clazz < classCount; clazz++) {
            for (int neuron = 0; neuron < hiddenCount; neuron++) {
                weights[outputWeightOffset(clazz) + neuron]
                        = random.nextGaussian() / Math.sqrt(hiddenCount);
            }
        }
    }

    // Calculate the hidden values and one output score for each label.
    private double[][] forward(double[] input) {
        double[] hiddenValues = new double[hiddenCount];
        double[] probabilities = new double[classCount];

        // First layer: each hidden neuron adds its weighted inputs and bias.
        // ReLU keeps positive results and changes negative results to zero.
        for (int neuron = 0; neuron < hiddenCount; neuron++) {
            int offset = neuron * (inputCount + 1);
            double sum = weights[offset + inputCount]; // bias is stored after the input weights
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
                sum += weights[offset + inputIndex] * input[inputIndex];
            }
            hiddenValues[neuron] = Math.max(0, sum);
        }

        // Output layer: each class combines the hidden values into one score.
        double largestScore = Double.NEGATIVE_INFINITY;
        for (int clazz = 0; clazz < classCount; clazz++) {
            int offset = outputWeightOffset(clazz);
            double score = weights[offset + hiddenCount]; // class bias
            for (int neuron = 0; neuron < hiddenCount; neuron++) {
                score += weights[offset + neuron] * hiddenValues[neuron];
            }
            probabilities[clazz] = score;
            largestScore = Math.max(largestScore, score);
        }

        // Softmax turns scores into probabilities. Subtracting the largest score
        // first keeps exponentials from overflowing without changing the answer.
        double total = 0;
        for (int clazz = 0; clazz < classCount; clazz++) {
            probabilities[clazz] = Math.exp(probabilities[clazz] - largestScore);
            total += probabilities[clazz];
        }
        for (int clazz = 0; clazz < classCount; clazz++) probabilities[clazz] /= total;
        return new double[][] {hiddenValues, probabilities};
    }

    /**
     * Returns the network's score for every label.
     *
     * @param input the number list to classify
     * @return scores in label order; they add to 1, but may not reflect true confidence
     */
    public double[] predict(double[] input) {
        validateInput(input);
        return forward(input)[1];
    }

    /**
     * Returns the number of weights and bias values in the network.
     *
     * @return the number of learned values
     */
    public long parameterCount() { return weights.length; }

    /**
     * Estimates bytes for the weights and Adam's two saved values per weight.
     * This does not include Java object overhead or temporary work space.
     *
     * @return estimated bytes used for the learned weights and their training state
     */
    public long trainingStateBytes() { return 24L * weights.length; }

    /**
     * Learns from labeled examples by repeatedly comparing guesses with answers.
     *
     * @param inputs one number list per example
     * @param targets one-hot labels: each row has a 1 for the correct class and 0 elsewhere
     * @param epochs number of times to train over all examples
     * @param rate size of each weight adjustment, from 0 to 1
     * @param batchSize number of examples used before making a weight adjustment
     */
    public void train(double[][] inputs, double[][] targets, int epochs, double rate, int batchSize) {
        if (inputs == null || targets == null || inputs.length == 0 || inputs.length != targets.length
                || epochs < 1 || batchSize < 1 || !Double.isFinite(rate) || rate <= 0 || rate > 1) {
            throw new IllegalArgumentException("Invalid training settings");
        }
        for (int example = 0; example < inputs.length; example++) {
            validateInput(inputs[example]);
            validateTarget(targets[example]);
        }

        // The order array lets each epoch see examples in a new shuffled order.
        int[] order = new int[inputs.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        double[] gradient = new double[weights.length];

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(order);
            for (int start = 0; start < order.length; start += batchSize) {
                Arrays.fill(gradient, 0);
                int end = Math.min(order.length, start + batchSize);

                // Add each example's gradient to the minibatch total.
                for (int position = start; position < end; position++) {
                    int example = order[position];
                    accumulateGradient(inputs[example], targets[example], gradient);
                }

                // Average the minibatch gradients, then make one Adam weight update.
                updateCount++;
                double correctedFirstMoment = 1 - Math.pow(ADAM_BETA_1, updateCount);
                double correctedSecondMoment = 1 - Math.pow(ADAM_BETA_2, updateCount);
                int batchExamples = end - start;
                for (int weight = 0; weight < weights.length; weight++) {
                    double averageGradient = gradient[weight] / batchExamples;
                    firstMoment[weight] = ADAM_BETA_1 * firstMoment[weight]
                            + (1 - ADAM_BETA_1) * averageGradient;
                    secondMoment[weight] = ADAM_BETA_2 * secondMoment[weight]
                            + (1 - ADAM_BETA_2) * averageGradient * averageGradient;

                    // Bias correction compensates for the averages starting at zero.
                    double first = firstMoment[weight] / correctedFirstMoment;
                    double second = secondMoment[weight] / correctedSecondMoment;
                    weights[weight] -= rate * first / (Math.sqrt(second) + ADAM_EPSILON);
                    if (!Double.isFinite(weights[weight])) {
                        throw new IllegalStateException("Non-finite Adam weight");
                    }
                }
            }
        }
    }

    // Find how one example's error should change each weight and add it to the batch.
    private void accumulateGradient(double[] input, double[] target, double[] gradient) {
        double[][] pass = forward(input);
        double[] hiddenValues = pass[0];
        double[] probabilities = pass[1];
        double[] hiddenErrors = new double[hiddenCount];

        // With softmax and cross-entropy, the output error is probability minus target.
        for (int clazz = 0; clazz < classCount; clazz++) {
            double outputError = probabilities[clazz] - target[clazz];
            int offset = outputWeightOffset(clazz);
            gradient[offset + hiddenCount] += outputError;
            for (int neuron = 0; neuron < hiddenCount; neuron++) {
                gradient[offset + neuron] += outputError * hiddenValues[neuron];
                hiddenErrors[neuron] += weights[offset + neuron] * outputError;
            }
        }

        // Carry each output error backward through ReLU to the first-layer weights.
        for (int neuron = 0; neuron < hiddenCount; neuron++) {
            if (hiddenValues[neuron] <= 0) continue; // ReLU's derivative is zero here.
            int offset = neuron * (inputCount + 1);
            gradient[offset + inputCount] += hiddenErrors[neuron];
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
                gradient[offset + inputIndex] += hiddenErrors[neuron] * input[inputIndex];
            }
        }
    }

    /**
     * Checks the calculated gradient against a small measured change in loss.
     * Used by {@link BackpropGradientCheck} to test the math.
     */
    double maximumGradientError(double[] input, double[] target) {
        double[] gradient = new double[weights.length];
        accumulateGradient(input, target, gradient);
        double maximumError = 0;
        double epsilon = 1e-6;
        for (int index = 0; index < weights.length; index++) {
            double original = weights[index];
            weights[index] = original + epsilon;
            double lossAbove = loss(input, target);
            weights[index] = original - epsilon;
            double lossBelow = loss(input, target);
            weights[index] = original;
            double numericalGradient = (lossAbove - lossBelow) / (2 * epsilon);
            maximumError = Math.max(maximumError, Math.abs(gradient[index] - numericalGradient));
        }
        return maximumError;
    }

    private double loss(double[] input, double[] target) {
        double[] probabilities = forward(input)[1];
        double total = 0;
        for (int clazz = 0; clazz < classCount; clazz++) {
            if (target[clazz] != 0) total -= Math.log(probabilities[clazz]);
        }
        return total;
    }

    private void validateTarget(double[] target) {
        if (target == null || target.length != classCount) {
            throw new IllegalArgumentException("Target dimensions do not match classes");
        }
        int ones = 0;
        for (double value : target) {
            if (value == 1) ones++;
            else if (value != 0) throw new IllegalArgumentException("One-hot target required");
        }
        if (ones != 1) throw new IllegalArgumentException("One-hot target required");
    }

    private void validateInput(double[] input) {
        if (input == null || input.length != inputCount) {
            throw new IllegalArgumentException("Input dimensions do not match the network");
        }
        for (double value : input) {
            if (!Double.isFinite(value) || value < -1 || value > 1) {
                throw new IllegalArgumentException("Inputs must be finite and in [-1,1]");
            }
        }
    }

    private int outputWeightOffset(int clazz) {
        return outputLayerOffset + clazz * (hiddenCount + 1);
    }

    private void shuffle(int[] values) {
        for (int i = values.length - 1; i > 0; i--) {
            int other = random.nextInt(i + 1);
            int saved = values[i];
            values[i] = values[other];
            values[other] = saved;
        }
    }
}
