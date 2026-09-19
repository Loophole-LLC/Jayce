// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

/**
 * Reads the MNIST image and label files used by the Java benchmark.
 *
 * <p>Each image is changed into one row of pixel numbers from 0 to 1. Each
 * digit label is changed into a row of ten values, with a 1 at the digit's
 * position. Files ending in {@code .gz} are unpacked automatically.
 */
public final class MnistLoader {

    private static final int IMAGE_MAGIC = 0x00000803;
    private static final int LABEL_MAGIC = 0x00000801;
    private static final int DIGIT_CLASSES = 10;

    private MnistLoader() {
    }

    /**
     * The images and labels after they have been read from the files.
     *
     * @param inputs one row of pixel numbers per image
     * @param targets one row of labels per image, with a 1 at the right digit
     * @param imageWidth image width in pixels
     * @param imageHeight image height in pixels
     */
    public record Dataset(double[][] inputs, double[][] targets, int imageWidth, int imageHeight) {
    }

    /**
     * Reads matching image and label files.
     *
     * @param imagesFile path to the MNIST image file
     * @param labelsFile path to its label file
     * @return images as number lists and labels as one-hot rows
     * @throws IOException if either file is missing, damaged, or does not match the MNIST format
     */
    public static Dataset load(Path imagesFile, Path labelsFile) throws IOException {
        double[][] inputs;
        int width;
        int height;

        try (DataInputStream in = new DataInputStream(openMaybeGzipped(imagesFile))) {
            int magic = in.readInt();
            if (magic != IMAGE_MAGIC) {
                throw new IOException("Not an MNIST image file (expected magic " + IMAGE_MAGIC
                        + ", got " + magic + "): " + imagesFile);
            }
            int count = in.readInt();
            height = in.readInt();
            width = in.readInt();

            inputs = new double[count][width * height];
            byte[] rawPixels = new byte[width * height];
            for (int n = 0; n < count; n++) {
                in.readFully(rawPixels);
                double[] pixels = inputs[n];
                for (int p = 0; p < rawPixels.length; p++) {
                    // Java bytes can be negative, so "& 0xFF" reads the file
                    // value as 0-255 before scaling it to the range 0-1.
                    pixels[p] = (rawPixels[p] & 0xFF) / 255.0;
                }
            }
        }

        int[] labels;
        try (DataInputStream in = new DataInputStream(openMaybeGzipped(labelsFile))) {
            int magic = in.readInt();
            if (magic != LABEL_MAGIC) {
                throw new IOException("Not an MNIST label file (expected magic " + LABEL_MAGIC
                        + ", got " + magic + "): " + labelsFile);
            }
            int count = in.readInt();
            labels = new int[count];
            for (int n = 0; n < count; n++) {
                labels[n] = in.readUnsignedByte();
            }
        }

        if (labels.length != inputs.length) {
            throw new IOException("Image count (" + inputs.length + ") does not match label count ("
                    + labels.length + ") — files may be mismatched or corrupt");
        }

        // Store a label as ten values. For digit 3, only position 3 is 1;
        // all other positions are 0. This tells the network the correct class.
        double[][] targets = new double[labels.length][DIGIT_CLASSES];
        for (int n = 0; n < labels.length; n++) {
            targets[n][labels[n]] = 1.0;
        }

        return new Dataset(inputs, targets, width, height);
    }

    private static InputStream openMaybeGzipped(Path file) throws IOException {
        InputStream raw = Files.newInputStream(file);
        String name = file.getFileName().toString();
        return name.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
    }
}
