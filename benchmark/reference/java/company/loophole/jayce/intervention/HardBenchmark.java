// Copyright (C) 2026 Loophole, LLC.
// SPDX-License-Identifier: AGPL-3.0-only
// See LICENSE.md in the repository root for terms and warranty information.

package company.loophole.jayce.intervention;

import company.loophole.jayce.MnistLoader;
import company.loophole.jayce.apm.JayceMemory;
import company.loophole.jayce.backprop.SoftmaxBackpropNetwork;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Runs the Java comparison between Jayce's APM learner and backpropagation.
 *
 * <p>It chooses settings using two validation seeds, then tests on ten other
 * seeds. Each test uses four data streams whose conditions change over twelve
 * phases. Both learners receive new examples only; neither uses replay.
 */
public final class HardBenchmark {
    enum Learner {
        JAYCE("jayce"), BACKPROP("backprop");
        final String label;
        Learner(String label) { this.label = label; }
        static Learner parse(String text) {
            for (Learner learner : values()) if (learner.label.equals(text)) return learner;
            throw new IllegalArgumentException("Unknown learner: " + text);
        }
    }

    record Config(int centers, double rate) {}
    record ModelStats(long fresh, long updates, long work, long bytes, long capacityBytes, int centers) {}
    record Checkpoint(int phase, int checkpoint, int condition, int[] correct, int n, ModelStats stats) {}
    record Trial(int[] correct, int n, ModelStats stats, double seconds, List<Checkpoint> checkpoints) {
        double accuracy() { return Arrays.stream(correct).average().orElseThrow() / n; }
    }
    record PreparedData(HardBenchmarkData.Data train, HardBenchmarkData.Data evaluation) {}
    record SeedAccuracy(long seed, double accuracy) {}
    record CellKey(String dataset, String budget, Learner learner) {}

    private static final int[] JAYCE_CENTERS = {12, 24, 48};
    private static final double[] JAYCE_RATES = {.001, .003, .01, .03, .1, .3, .6, 1};
    private static final double[] BACKPROP_RATES = {.0001, .0003, .001, .003, .01, .03};
    private static final long[] VALIDATION_SEEDS = {213271, 214281};
    private static final long FIRST_TEST_SEED = 2280101;
    private static final long TEST_SEED_STEP = 104729;
    private static final int TEST_SEEDS = 10;
    private static final double T_95_DF_9 = 2.262157;

    private final int dimensions;
    private final int classes;
    private final Learner learner;
    private final JayceMemory jayce;
    private final SoftmaxBackpropNetwork backprop;
    private final double rate;
    private long fresh;
    private long adamWork;
    private long adamUpdates;

    private HardBenchmark(int dimensions, int classes, Learner learner, Config config, long seed) {
        this.dimensions = dimensions;
        this.classes = classes;
        this.learner = learner;
        this.rate = config.rate;
        if (learner == Learner.JAYCE) {
            jayce = new JayceMemory(dimensions, classes, config.centers, .1);
            backprop = null;
        } else {
            jayce = null;
            int hidden = dimensions == 64 ? 32 : 96;
            backprop = new SoftmaxBackpropNetwork(dimensions, hidden, classes, seed);
        }
    }

    private void train(HardBenchmarkData.Data batch) {
        if (learner == Learner.JAYCE) {
            int[] labels = new int[batch.y().length];
            for (int i = 0; i < labels.length; i++) labels[i] = HardBenchmarkData.argmax(batch.y()[i]);
            jayce.train(batch.x(), labels, rate);
        } else {
            // Train on this batch and let Adam adjust the weights once.
            backprop.train(batch.x(), batch.y(), 1, rate, batch.x().length);
            adamWork += adamWork(batch.x().length, dimensions, classes, backprop.parameterCount());
            adamUpdates++;
        }
        fresh += batch.x().length;
    }

    private double[] predict(double[] input) {
        return learner == Learner.JAYCE ? jayce.scores(input) : backprop.predict(input);
    }

    private ModelStats stats() {
        if (learner == Learner.JAYCE) {
            var s = jayce.statistics();
            return new ModelStats(s.examples(), s.updates(), s.work(), jayce.persistentBytes(),
                    jayce.capacityBytes(), s.activePrototypes());
        }
        long bytes = backprop.trainingStateBytes();
        return new ModelStats(fresh, adamUpdates, adamWork, bytes, bytes, 0);
    }

    private long batchWorkUpperBound() {
        if (learner == Learner.JAYCE) {
            return jayce.batchWorkUpperBound(HardBenchmarkData.BATCH_SIZE);
        }
        return adamWork(HardBenchmarkData.BATCH_SIZE, dimensions, classes, backprop.parameterCount());
    }

    private static long adamWork(int examples, int dimensions, int classes, long parameters) {
        int hidden = dimensions == 64 ? 32 : 96;
        long edges = (long) dimensions * hidden + (long) hidden * classes;
        return examples * (2 * edges + 2L * hidden + classes + (long) hidden * classes + 5)
                + 16 * parameters;
    }

    private static int dimensionsFor(String dataset) {
        return dataset.startsWith("shapes") ? 64 : 196;
    }

    private static int classesFor(String dataset) {
        return dataset.startsWith("shapes") ? 4 : 10;
    }

    private static int hiddenFor(int dimensions) {
        return dimensions == 64 ? 32 : 96;
    }

    // This estimates the memory used by weights and Adam's two saved values per weight.
    private static long backpropStateBytes(int dimensions, int classes) {
        int hidden = hiddenFor(dimensions);
        long parameters = (long) hidden * (dimensions + 1L) + (long) classes * (hidden + 1L);
        return 24L * parameters;
    }

    private static long jayceCapacityBytes(int dimensions, int classes, int centersPerClass) {
        return 8L * dimensions * classes * centersPerClass + 4L * classes;
    }

    private static long memoryLimitBytes(String dataset) {
        return backpropStateBytes(dimensionsFor(dataset), classesFor(dataset));
    }

    private static long capacityBytes(Learner learner, String dataset, Config config) {
        if (learner == Learner.JAYCE) {
            return jayceCapacityBytes(dimensionsFor(dataset), classesFor(dataset), config.centers);
        }
        return memoryLimitBytes(dataset);
    }

    private static List<Config> candidates(Learner learner, String dataset, String budget) {
        List<Config> result = new ArrayList<>();
        if (learner == Learner.JAYCE) {
            List<Integer> centerCounts = new ArrayList<>();
            for (int centers : JAYCE_CENTERS) centerCounts.add(centers);
            if (budget.equals("memory")) {
                int dimensions = dimensionsFor(dataset);
                int classes = classesFor(dataset);
                int affordable = (int) ((memoryLimitBytes(dataset) - 4L * classes)
                        / (8L * dimensions * classes));
                if (affordable > 0 && !centerCounts.contains(affordable)) centerCounts.add(affordable);
            }
            for (int centers : centerCounts) {
                if (budget.equals("memory") && jayceCapacityBytes(
                        dimensionsFor(dataset), classesFor(dataset), centers) > memoryLimitBytes(dataset)) continue;
                for (double rate : JAYCE_RATES) result.add(new Config(centers, rate));
            }
        } else {
            for (double rate : BACKPROP_RATES) result.add(new Config(0, rate));
        }
        return result;
    }

    private static Config choose(Map<Config, Double> scores, List<Config> candidates) {
        Config best = candidates.get(0);
        for (Config candidate : candidates) {
            if (scores.get(candidate) > scores.get(best) + 1e-12) best = candidate;
        }
        return best;
    }

    private static Trial run(Learner learner, Config config, long seed, String budget,
                             String dataset, HardBenchmarkData.ShapeStream shapeStream,
                             HardBenchmarkData.ImageStream imageStream, boolean detailed,
                             boolean useFullEvaluation) {
        int dimensions = dataset.startsWith("shapes") ? 64 : 196;
        int classes = dataset.startsWith("shapes") ? 4 : 10;
        HardBenchmarkData.Data[] full;
        HardBenchmarkData.Data[] checkpoints;
        long phaseWork;
        if (shapeStream != null) {
            full = shapeStream.evaluation;
            checkpoints = full;
            phaseWork = shapeStream.phaseWork;
        } else {
            full = imageStream.evaluation.full;
            checkpoints = imageStream.evaluation.checkpoints;
            phaseWork = imageStream.phaseWork;
        }

        HardBenchmark model = new HardBenchmark(dimensions, classes, learner, config, seed);
        List<Checkpoint> points = new ArrayList<>();
        double seconds = 0;
        for (int phase = 0; phase < HardBenchmarkData.PHASE_ORDER.length; phase++) {
            long workAtStart = model.stats().work;
            int batchNumber = 0;
            if (detailed) points.add(evaluate(model, checkpoints, phase, 0, -1));
            for (int checkpoint = 1; checkpoint <= 2; checkpoint++) {
                int fraction = checkpoint == 1 ? 1 : 4;
                while (budget.equals("work")
                        ? model.stats().work - workAtStart + model.batchWorkUpperBound() <= phaseWork * fraction / 4
                        : batchNumber < HardBenchmarkData.BATCHES_PER_PHASE * fraction / 4) {
                    HardBenchmarkData.Data batch = shapeStream != null
                            ? shapeStream.batch(phase, batchNumber)
                            : imageStream.batch(phase, batchNumber);
                    long before = model.stats().work;
                    long bound = model.batchWorkUpperBound();
                    long start = System.nanoTime();
                    model.train(batch);
                    seconds += (System.nanoTime() - start) / 1_000_000_000.0;
                    if (model.stats().work - before > bound) {
                        throw new IllegalStateException("Training exceeded its precomputed work bound");
                    }
                    batchNumber++;
                }
                if (detailed) points.add(evaluate(model, checkpoints, phase, checkpoint,
                        HardBenchmarkData.PHASE_ORDER[phase]));
            }
        }

        ModelStats beforeEvaluation = model.stats();
        HardBenchmarkData.Data[] finalEvaluation = useFullEvaluation ? full : checkpoints;
        int[] correct = correct(model, finalEvaluation, -1);
        if (detailed && !beforeEvaluation.equals(model.stats())) {
            throw new IllegalStateException("Evaluation changed model state");
        }
        return new Trial(correct, finalEvaluation[0].x().length, beforeEvaluation, seconds, points);
    }

    private static Checkpoint evaluate(HardBenchmark model, HardBenchmarkData.Data[] data,
                                       int phase, int checkpoint, int currentCondition) {
        ModelStats before = model.stats();
        int[] counts = correct(model, data, checkpoint == 1 ? currentCondition : -1);
        if (!before.equals(model.stats())) throw new IllegalStateException("Checkpoint evaluation changed model state");
        return new Checkpoint(phase, checkpoint, HardBenchmarkData.PHASE_ORDER[phase], counts,
                data[0].x().length, before);
    }

    private static int[] correct(HardBenchmark model, HardBenchmarkData.Data[] data, int currentOnly) {
        int[] counts = new int[data.length];
        Arrays.fill(counts, -1);
        for (int condition = 0; condition < data.length; condition++) {
            if (currentOnly >= 0 && currentOnly != condition) continue;
            counts[condition] = 0;
            for (int i = 0; i < data[condition].x().length; i++) {
                if (HardBenchmarkData.argmax(model.predict(data[condition].x()[i]))
                        == HardBenchmarkData.argmax(data[condition].y()[i])) counts[condition]++;
            }
        }
        return counts;
    }

    private static PreparedData loadData(String dataset, boolean test) throws IOException {
        if (dataset.startsWith("shapes")) return new PreparedData(null, null);
        String folder = dataset.startsWith("mnist") ? "mnist" : "fashion-mnist";
        Path raw = Path.of("data", folder, "raw");
        MnistLoader.Dataset original = MnistLoader.load(
                raw.resolve("train-images-idx3-ubyte.gz"), raw.resolve("train-labels-idx1-ubyte.gz"));
        if (original.inputs().length != 60_000) throw new IOException("Expected 60,000 training images");
        HardBenchmarkData.Data train = HardBenchmarkData.reduce(original, 0, 50_000);
        HardBenchmarkData.Data evaluation;
        if (test) {
            MnistLoader.Dataset heldOut = MnistLoader.load(
                    raw.resolve("t10k-images-idx3-ubyte.gz"), raw.resolve("t10k-labels-idx1-ubyte.gz"));
            if (heldOut.inputs().length != 10_000) throw new IOException("Expected 10,000 held-out images");
            evaluation = HardBenchmarkData.reduce(heldOut, 0, 10_000);
        } else {
            evaluation = HardBenchmarkData.reduce(original, 50_000, 60_000);
        }
        return new PreparedData(train, evaluation);
    }

    private static HardBenchmarkData.ShapeStream shapeStream(String dataset, long seed) {
        return new HardBenchmarkData.ShapeStream(dataset, seed);
    }

    private static HardBenchmarkData.ImageStream imageStream(long seed, PreparedData data) {
        return new HardBenchmarkData.ImageStream(seed, data.train,
                new HardBenchmarkData.ImageEvaluation(data.evaluation, 50));
    }

    private static void tune(Path output) throws IOException {
        requireNewOutput(output, "validation.csv", "selected.csv", "freeze.sha256", "source.zip");
        String sourceHash = snapshotSources(output.resolve("source.zip"));
        Map<CellKey, Map<Config, Double>> scores = new LinkedHashMap<>();
        try (var csv = Files.newBufferedWriter(output.resolve("validation.csv"), StandardCharsets.UTF_8)) {
            csv.write("dataset,budget,learner,centers_per_class,learning_rate,seed,accuracy,work,state_bytes,capacity_bytes\n");
            for (String dataset : HardBenchmarkData.DATASETS) {
                PreparedData data = loadData(dataset, false);
                for (String budget : HardBenchmarkData.BUDGETS) {
                    for (Learner learner : Learner.values()) {
                        CellKey key = new CellKey(dataset, budget, learner);
                        List<Config> candidateSet = candidates(learner, dataset, budget);
                        Map<Config, Double> means = new LinkedHashMap<>();
                        scores.put(key, means);
                        for (long seed : VALIDATION_SEEDS) {
                            HardBenchmarkData.ShapeStream shapes = dataset.startsWith("shapes")
                                    ? shapeStream(dataset, seed) : null;
                            HardBenchmarkData.ImageStream images = shapes == null ? imageStream(seed, data) : null;
                            for (Config config : candidateSet) {
                                Trial trial = run(learner, config, seed, budget, dataset, shapes, images,
                                        false, false);
                                means.merge(config, trial.accuracy() / VALIDATION_SEEDS.length, Double::sum);
                                csv.write(String.format(Locale.ROOT, "%s,%s,%s,%d,%.4f,%d,%.8f,%d,%d,%d%n",
                                        dataset, budget, learner.label, config.centers, config.rate, seed,
                                        trial.accuracy(), trial.stats.work, trial.stats.bytes, trial.stats.capacityBytes));
                            }
                            csv.flush();
                        }
                        System.out.println("Validation complete: " + dataset + " / " + budget + " / " + learner.label);
                    }
                }
                data = null;
                System.gc();
            }
        }

        try (var csv = Files.newBufferedWriter(output.resolve("selected.csv"), StandardCharsets.UTF_8)) {
            csv.write("dataset,budget,learner,centers_per_class,learning_rate,validation_accuracy,memory_limit_bytes,capacity_bytes\n");
            for (String dataset : HardBenchmarkData.DATASETS) {
                for (String budget : HardBenchmarkData.BUDGETS) {
                    for (Learner learner : Learner.values()) {
                        CellKey key = new CellKey(dataset, budget, learner);
                        List<Config> candidateSet = candidates(learner, dataset, budget);
                        Config best = choose(scores.get(key), candidateSet);
                        String limit = budget.equals("memory") ? Long.toString(memoryLimitBytes(dataset)) : "";
                        csv.write(String.format(Locale.ROOT, "%s,%s,%s,%d,%.4f,%.8f,%s,%d%n",
                                dataset, budget, learner.label, best.centers, best.rate,
                                scores.get(key).get(best), limit, capacityBytes(learner, dataset, best)));
                    }
                }
            }
        }
        writeFreeze(output, sourceHash);
        System.out.println("Validation choices frozen in " + output.resolve("selected.csv"));
    }

    private static void test(Path output) throws IOException {
        verifyFreeze(output);
        requireNewOutput(output, "checkpoints.csv", "final.csv", "summary.csv");
        Map<CellKey, Config> selected = readSelected(output.resolve("selected.csv"));
        List<FinalRow> rows = new ArrayList<>();
        try (var checkpoints = Files.newBufferedWriter(output.resolve("checkpoints.csv"), StandardCharsets.UTF_8);
             var finals = Files.newBufferedWriter(output.resolve("final.csv"), StandardCharsets.UTF_8)) {
            checkpoints.write("seed,dataset,budget,learner,centers_per_class,learning_rate,phase,checkpoint,current_condition,correct0,correct1,correct2,correct3,n,fresh,updates,work,state_bytes,capacity_bytes,active_centers\n");
            finals.write("seed,dataset,budget,learner,centers_per_class,learning_rate,correct0,correct1,correct2,correct3,n,fresh,updates,work,state_bytes,capacity_bytes,active_centers,seconds\n");
            for (String dataset : HardBenchmarkData.DATASETS) {
                PreparedData data = loadData(dataset, true);
                for (int seedIndex = 0; seedIndex < TEST_SEEDS; seedIndex++) {
                    long seed = FIRST_TEST_SEED + TEST_SEED_STEP * seedIndex;
                    HardBenchmarkData.ShapeStream shapes = dataset.startsWith("shapes")
                            ? shapeStream(dataset, seed) : null;
                    HardBenchmarkData.ImageStream images = shapes == null ? imageStream(seed, data) : null;
                    for (String budget : HardBenchmarkData.BUDGETS) {
                        for (Learner learner : Learner.values()) {
                            CellKey key = new CellKey(dataset, budget, learner);
                            Config config = selected.get(key);
                            Trial trial = run(learner, config, seed, budget, dataset, shapes, images,
                                    true, true);
                            for (Checkpoint point : trial.checkpoints) {
                                checkpoints.write(checkpointCsv(seed, dataset, budget, learner, config, point));
                            }
                            FinalRow row = new FinalRow(seed, dataset, budget, learner, config,
                                    trial.correct, trial.n, trial.stats, trial.seconds);
                            rows.add(row);
                            finals.write(finalCsv(row));
                            finals.flush();
                        }
                    }
                    System.out.println("Held-out test complete: " + dataset + " / seed " + seed);
                }
                data = null;
                System.gc();
            }
        }
        writeSummary(output.resolve("summary.csv"), rows);
        System.out.println("Java benchmark results written to " + output.toAbsolutePath());
    }

    private record FinalRow(long seed, String dataset, String budget, Learner learner, Config config,
                            int[] correct, int n, ModelStats stats, double seconds) {
        double accuracy() { return Arrays.stream(correct).average().orElseThrow() / n; }
    }

    private static String checkpointCsv(long seed, String dataset, String budget, Learner learner,
                                        Config config, Checkpoint point) {
        ModelStats s = point.stats;
        return String.format(Locale.ROOT,
                "%d,%s,%s,%s,%d,%.4f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d%n",
                seed, dataset, budget, learner.label, config.centers, config.rate, point.phase,
                point.checkpoint, point.condition, point.correct[0], point.correct[1], point.correct[2],
                point.correct[3], point.n, s.fresh, s.updates, s.work, s.bytes, s.capacityBytes, s.centers);
    }

    private static String finalCsv(FinalRow row) {
        ModelStats s = row.stats;
        return String.format(Locale.ROOT,
                "%d,%s,%s,%s,%d,%.4f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f%n",
                row.seed, row.dataset, row.budget, row.learner.label, row.config.centers,
                row.config.rate, row.correct[0], row.correct[1], row.correct[2], row.correct[3],
                row.n, s.fresh, s.updates, s.work, s.bytes, s.capacityBytes, s.centers, row.seconds);
    }

    private static void writeSummary(Path path, List<FinalRow> rows) throws IOException {
        Map<CellKey, List<FinalRow>> grouped = new LinkedHashMap<>();
        for (String dataset : HardBenchmarkData.DATASETS) {
            for (String budget : HardBenchmarkData.BUDGETS) {
                for (Learner learner : Learner.values()) grouped.put(new CellKey(dataset, budget, learner), new ArrayList<>());
            }
        }
        for (FinalRow row : rows) grouped.get(new CellKey(row.dataset, row.budget, row.learner)).add(row);
        try (var csv = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            csv.write("dataset,budget,learner,seeds,accuracy_mean_percent,accuracy_sd_pp,state_kib_mean,capacity_kib_mean,memory_limit_kib,work_millions_mean,jayce_minus_backprop_pp,paired_95_low_pp,paired_95_high_pp,jayce_ahead_seeds\n");
            for (String dataset : HardBenchmarkData.DATASETS) {
                for (String budget : HardBenchmarkData.BUDGETS) {
                    List<FinalRow> jayce = grouped.get(new CellKey(dataset, budget, Learner.JAYCE));
                    List<FinalRow> backprop = grouped.get(new CellKey(dataset, budget, Learner.BACKPROP));
                    double[] differences = new double[jayce.size()];
                    int ahead = 0;
                    for (int i = 0; i < jayce.size(); i++) {
                        differences[i] = 100 * (jayce.get(i).accuracy() - backprop.get(i).accuracy());
                        if (differences[i] > 0) ahead++;
                    }
                    double meanDifference = mean(differences);
                    double halfWidth = T_95_DF_9 * sampleSd(differences) / Math.sqrt(differences.length);
                    for (Learner learner : Learner.values()) {
                        List<FinalRow> group = grouped.get(new CellKey(dataset, budget, learner));
                        double[] accuracies = group.stream().mapToDouble(r -> r.accuracy() * 100).toArray();
                        double[] states = group.stream().mapToDouble(r -> r.stats.bytes / 1024.0).toArray();
                        double[] capacities = group.stream().mapToDouble(r -> r.stats.capacityBytes / 1024.0).toArray();
                        double[] work = group.stream().mapToDouble(r -> r.stats.work / 1_000_000.0).toArray();
                        String limit = budget.equals("memory")
                                ? String.format(Locale.ROOT, "%.2f", memoryLimitBytes(dataset) / 1024.0) : "";
                        csv.write(String.format(Locale.ROOT, "%s,%s,%s,%d,%.4f,%.4f,%.2f,%.2f,%s,%.4f,%.4f,%.4f,%.4f,%d%n",
                                dataset, budget, learner.label, group.size(), mean(accuracies), sampleSd(accuracies),
                                mean(states), mean(capacities), limit, mean(work), meanDifference,
                                meanDifference - halfWidth, meanDifference + halfWidth, ahead));
                    }
                }
            }
        }
    }

    private static double mean(double[] values) {
        return Arrays.stream(values).average().orElse(0);
    }
    private static double sampleSd(double[] values) {
        if (values.length < 2) return 0;
        double mean = mean(values), sum = 0;
        for (double value : values) sum += (value - mean) * (value - mean);
        return Math.sqrt(sum / (values.length - 1));
    }

    private static Map<CellKey, Config> readSelected(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        if (lines.size() != 1 + HardBenchmarkData.DATASETS.length * HardBenchmarkData.BUDGETS.length * Learner.values().length) {
            throw new IOException("Selection file is incomplete");
        }
        Map<CellKey, Config> selected = new LinkedHashMap<>();
        for (String line : lines.subList(1, lines.size())) {
            String[] columns = line.split(",");
            if (columns.length != 8) throw new IOException("Malformed selected configuration");
            CellKey key = new CellKey(columns[0], columns[1], Learner.parse(columns[2]));
            Config config = new Config(Integer.parseInt(columns[3]), Double.parseDouble(columns[4]));
            long limit = columns[6].isBlank() ? 0 : Long.parseLong(columns[6]);
            long capacity = Long.parseLong(columns[7]);
            boolean memoryBudget = key.budget.equals("memory");
            if (!candidates(key.learner, key.dataset, key.budget).contains(config)
                    || capacity != capacityBytes(key.learner, key.dataset, config)
                    || (memoryBudget && limit != memoryLimitBytes(key.dataset))
                    || (memoryBudget && capacity > limit)
                    || (!memoryBudget && limit != 0)
                    || selected.put(key, config) != null) {
                throw new IOException("Invalid or duplicated selected configuration: " + line);
            }
        }
        for (String dataset : HardBenchmarkData.DATASETS) {
            for (String budget : HardBenchmarkData.BUDGETS) {
                for (Learner learner : Learner.values()) {
                    if (!selected.containsKey(new CellKey(dataset, budget, learner))) {
                        throw new IOException("Missing selection for " + dataset + "/" + budget + "/" + learner.label);
                    }
                }
            }
        }
        return selected;
    }

    private static void writeFreeze(Path output, String sourceHash) throws IOException {
        if (!sourceHash.equals(sha256Sources(Path.of("reference/java")))) {
            throw new IOException("Java source changed during tuning; start a new run");
        }
        Map<String, String> values = new LinkedHashMap<>();
        values.put("source_sha256", sourceHash);
        values.put("source.zip", sha256(output.resolve("source.zip")));
        values.put("selected_sha256", sha256(output.resolve("selected.csv")));
        for (Path dataFile : benchmarkDataFiles()) values.put(dataFile.toString(), sha256(dataFile));
        try (var writer = Files.newBufferedWriter(output.resolve("freeze.sha256"), StandardCharsets.UTF_8)) {
            for (var entry : values.entrySet()) writer.write(entry.getKey() + "=" + entry.getValue() + "\n");
        }
    }

    private static void verifyFreeze(Path output) throws IOException {
        Path freeze = output.resolve("freeze.sha256");
        if (!Files.isRegularFile(freeze) || !Files.isRegularFile(output.resolve("selected.csv"))) {
            throw new IOException("Run 'tune' first; test requires the frozen selection and checksums");
        }
        Map<String, String> recorded = new HashMap<>();
        for (String line : Files.readAllLines(freeze, StandardCharsets.UTF_8)) {
            int split = line.indexOf('=');
            if (split > 0) recorded.put(line.substring(0, split), line.substring(split + 1));
        }
        requireHash(recorded, "source_sha256", sha256Sources(Path.of("reference/java")));
        requireHash(recorded, "source.zip", sha256(output.resolve("source.zip")));
        requireHash(recorded, "selected_sha256", sha256(output.resolve("selected.csv")));
        for (Path dataFile : benchmarkDataFiles()) requireHash(recorded, dataFile.toString(), sha256(dataFile));
    }

    private static void requireHash(Map<String, String> expected, String key, String actual) throws IOException {
        if (!actual.equals(expected.get(key))) throw new IOException("Frozen hash mismatch for " + key);
    }

    private static List<Path> benchmarkDataFiles() {
        List<Path> result = new ArrayList<>();
        for (String folder : List.of("mnist", "fashion-mnist")) {
            Path raw = Path.of("data", folder, "raw");
            for (String name : List.of("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")) result.add(raw.resolve(name));
        }
        return result;
    }

    // Keep the exact source alongside the results, independent of Git history.
    private static String snapshotSources(Path archive) throws IOException {
        Path root = Path.of("reference/java");
        try (var zip = new ZipOutputStream(Files.newOutputStream(archive))) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (Path file : sourceFiles(root)) {
                String relative = root.relativize(file).toString().replace('\\', '/');
                byte[] bytes = Files.readAllBytes(file);
                digest.update(relative.getBytes(StandardCharsets.UTF_8));
                digest.update(bytes);
                var entry = new ZipEntry("reference/java/" + relative);
                entry.setTime(0);
                zip.putNextEntry(entry);
                zip.write(bytes);
                zip.closeEntry();
            }
            return toHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("Every Java runtime includes SHA-256", impossible);
        }
    }

    private static List<Path> sourceFiles(Path root) throws IOException {
        try (var paths = Files.walk(root)) {
            return paths.filter(path -> path.toString().endsWith(".java")).sorted().toList();
        }
    }

    private static String sha256Sources(Path root) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (Path file : sourceFiles(root)) {
                digest.update(root.relativize(file).toString().replace('\\', '/').getBytes(StandardCharsets.UTF_8));
                digest.update(Files.readAllBytes(file));
            }
            return toHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("Every Java runtime includes SHA-256", impossible);
        }
    }

    private static String sha256(Path path) throws IOException {
        try {
            return toHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("Every Java runtime includes SHA-256", impossible);
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format("%02x", value));
        return result.toString();
    }

    private static void requireNewOutput(Path output, String... files) throws IOException {
        Files.createDirectories(output);
        for (String name : files) {
            if (Files.exists(output.resolve(name))) throw new IOException("Output already exists: " + output.resolve(name));
        }
    }

    private static void smoke() {
        for (String dataset : HardBenchmarkData.DATASETS) {
            long limit = memoryLimitBytes(dataset);
            if (capacityBytes(Learner.BACKPROP, dataset, new Config(0, .003)) != limit) {
                throw new IllegalStateException("Backprop state differs from its memory limit");
            }
            for (Config config : candidates(Learner.JAYCE, dataset, "memory")) {
                if (capacityBytes(Learner.JAYCE, dataset, config) > limit) {
                    throw new IllegalStateException("Jayce configuration exceeds its memory limit");
                }
            }
        }
        HardBenchmarkData.ShapeStream stream = new HardBenchmarkData.ShapeStream("shapes-mixed", 91);
        for (Learner learner : Learner.values()) {
            Config config = learner == Learner.JAYCE ? new Config(12, .1) : new Config(0, .003);
            HardBenchmark model = new HardBenchmark(64, 4, learner, config, 91);
            model.train(stream.batch(0, 0));
            double[] probabilities = model.predict(stream.batch(0, 1).x()[0]);
            double total = Arrays.stream(probabilities).sum();
            if (!Double.isFinite(total) || Math.abs(total - 1) > 1e-9 || model.stats().fresh != 16) {
                throw new IllegalStateException("Java benchmark smoke test failed for " + learner.label);
            }
            System.out.printf(Locale.ROOT, "%s: fresh=%d, work=%d, probability-sum=%.6f%n",
                    learner.label, model.stats().fresh, model.stats().work, total);
        }
    }

    /**
     * Runs the requested benchmark step.
     *
     * @param args command-line arguments that choose the step and output folder
     * @throws Exception if the selected step cannot read or write its files
     */
    public static void main(String[] args) throws Exception {
        System.setProperty("java.awt.headless", "true");
        if (args.length == 1 && args[0].equals("smoke")) {
            smoke();
            return;
        }
        if (args.length != 2 || !(args[0].equals("tune") || args[0].equals("test"))) {
            throw new IllegalArgumentException("Usage: HardBenchmark smoke | tune|test output-directory");
        }
        Path output = Path.of(args[1]);
        if (args[0].equals("tune")) tune(output);
        else test(output);
    }
}
