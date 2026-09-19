// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Random;

/**
 * Draws four simple black-and-white shapes for the Java benchmark.
 *
 * <p>The shapes are a rectangle, a cross, a diamond, and stripes. The
 * benchmark can turn them into number lists and give the same examples to
 * both learners without downloading a picture dataset.
 */
public final class ShapePatterns {

    /** Number of different shapes this class can draw. */
    public static final int CLASS_COUNT = 4;
    /** Names for shape numbers 0 through 3. */
    public static final String[] NAMES = {"rectangle", "cross", "diamond", "stripes"};

    private ShapePatterns() {
    }

    /**
     * Draws one of the four shapes without shifting it.
     *
     * @param shapeClass shape number from 0 to 3
     * @param size width and height of the square image
     * @return the drawn shape
     */
    public static BufferedImage draw(int shapeClass, int size) {
        return draw(shapeClass, size, null);
    }

    /**
     * Draws one of the four shapes, optionally shifted by up to one pixel.
     *
     * @param shapeClass shape number from 0 to 3
     * @param size width and height of the square image
     * @param jitterRandom random-number source for the small shift, or {@code null} for no shift
     * @return the drawn shape
     */
    public static BufferedImage draw(int shapeClass, int size, Random jitterRandom) {
        int jitterX = jitterRandom == null ? 0 : jitterRandom.nextInt(3) - 1;
        int jitterY = jitterRandom == null ? 0 : jitterRandom.nextInt(3) - 1;

        BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, size, size);
        g.setColor(Color.BLACK);

        switch (shapeClass) {
            case 0 -> g.fillRect(1 + jitterX, 2 + jitterY, size - 2, size - 4); // filled rectangle
            case 1 -> {
                g.setStroke(new BasicStroke(2.5f));
                g.drawLine(jitterX, jitterY, size - 1 + jitterX, size - 1 + jitterY);
                g.drawLine(size - 1 + jitterX, jitterY, jitterX, size - 1 + jitterY);
            }
            case 2 -> {
                int mid = size / 2;
                g.fillPolygon(
                        new int[] {mid + jitterX, size - 1 + jitterX, mid + jitterX, jitterX},
                        new int[] {jitterY, mid + jitterY, size - 1 + jitterY, mid + jitterY}, 4);
            }
            case 3 -> {
                for (int y = Math.floorMod(jitterY, 2); y < size; y += 2) {
                    g.fillRect(0, y, size, 1);
                }
            }
            default -> throw new IllegalArgumentException("Unknown shape class " + shapeClass);
        }
        g.dispose();
        return image;
    }
}
