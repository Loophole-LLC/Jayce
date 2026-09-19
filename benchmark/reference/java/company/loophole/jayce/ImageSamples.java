// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Turns an image into a list of pixel numbers that a classifier can use.
 *
 * <p>An image is a grid of pixels. This class changes it to one long list,
 * going across each row before moving to the next. The result uses the same
 * 0-to-1 grayscale values as {@link MnistLoader}.
 */
public final class ImageSamples {

    private ImageSamples() {
    }

    /**
     * Reads an image, resizes it, and turns it into a row-by-row list of
     * grayscale values from 0 (black) to 1 (white).
     *
     * @param imageFile image file to read
     * @param width width of the resized image in pixels
     * @param height height of the resized image in pixels
     * @return one number per pixel, from left to right and top to bottom
     * @throws IOException if the image cannot be read
     */
    public static double[] loadGrayscale(Path imageFile, int width, int height) throws IOException {
        BufferedImage original = ImageIO.read(imageFile.toFile());
        if (original == null) {
            throw new IOException("Could not decode image (unsupported or corrupt file): " + imageFile);
        }
        return toPixelVector(original, width, height);
    }

    /**
     * Converts an image already loaded in memory to grayscale pixel numbers.
     *
     * @param image image to convert
     * @param width width of the resized image in pixels
     * @param height height of the resized image in pixels
     * @return one number per pixel, from left to right and top to bottom
     */
    public static double[] toPixelVector(BufferedImage image, int width, int height) {
        BufferedImage resized = resizeTo(image, width, height);

        double[] pixels = new double[width * height];
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[index++] = toGrayscale(resized.getRGB(x, y));
            }
        }
        return pixels;
    }

    /**
     * Turns a list of grayscale pixel numbers back into an image.
     *
     * @param grayscale one number per pixel, from 0 (black) to 1 (white)
     * @param width image width in pixels
     * @param height image height in pixels
     * @return the grayscale image
     */
    public static BufferedImage toImage(double[] grayscale, int width, int height) {
        if (grayscale.length != width * height) {
            throw new IllegalArgumentException("Expected " + (width * height) + " pixels, got " + grayscale.length);
        }
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int level = (int) Math.round(clamp01(grayscale[index++]) * 255);
                int rgb = (level << 16) | (level << 8) | level;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static BufferedImage resizeTo(BufferedImage source, int width, int height) {
        if (source.getWidth() == width && source.getHeight() == height) {
            return source;
        }
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        // Blend nearby pixels when resizing so small images look smoother.
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(source, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private static double toGrayscale(int argb) {
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8) & 0xFF;
        int b = argb & 0xFF;
        // Give green more weight because people see green brightness more clearly.
        double luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        return luminance / 255.0;
    }
}
