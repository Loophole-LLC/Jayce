// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce.backprop;

/** Checks that backpropagation's calculated weight changes agree with measured loss changes. */
public final class BackpropGradientCheck {
    private BackpropGradientCheck() {}

    /**
     * Runs the gradient check and prints the largest difference it found.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        SoftmaxBackpropNetwork model = new SoftmaxBackpropNetwork(2, 3, 2, 17);
        double error = model.maximumGradientError(new double[] {0.2, -0.3}, new double[] {1, 0});
        if (!Double.isFinite(error) || error > 1e-5) {
            throw new IllegalStateException("Backprop gradient check failed: max error=" + error);
        }
        System.out.printf("Backprop gradient check passed (maximum error %.3g).%n", error);
    }
}
