# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""Adaptive next-byte prototypes with shared question exemplars.

Each prototype belongs to one answer prefix and predicts one byte. Its question
centroid can cover several wordings. Question exemplars are stored once, shared
across all answer positions, so a correction can detach one member and recenter
its old cluster without changing the answers taught to other questions.
"""

from __future__ import annotations

import numpy as np

from jayce_tokens import END, Match, MemoryFull, PrototypeError, TokenMemory, unit

QUESTION_DIMENSIONS = 384
PREFIX_DIMENSIONS = 128
DIMENSIONS = QUESTION_DIMENSIONS + PREFIX_DIMENSIONS
MIN_SIMILARITY = 0.84
MIN_MARGIN = 0.05
JOIN_SIMILARITY = 0.78
SCALE = np.float32(np.sqrt(2))


def question_key(vector):
    # The final question block fingerprints the original wording. It has only
    # 2% of similarity weight, but distinguishes exact exemplars for corrections.
    return np.packbits(vector[QUESTION_DIMENSIONS - 64:QUESTION_DIMENSIONS] > 0).tobytes()


def prefix_key(vector):
    return np.packbits(vector[QUESTION_DIMENSIONS:] > 0).tobytes()


class PatternMemory(TokenMemory):
    """Bounded, local centroid learning; no parent features or answer lookup."""

    def __init__(self, rate=0.1, max_prototypes=4096):
        super().__init__(rate, max_prototypes)
        self.samples = np.empty((0, QUESTION_DIMENSIONS), dtype=np.float32)
        self.members: list[dict[int, int]] = []
        self._index()

    def _index(self):
        self._questions = {question_key(row): i for i, row in enumerate(self.samples)}
        self._prefixes = {}
        for i, row in enumerate(self.vectors):
            self._prefixes.setdefault(prefix_key(row), []).append(i)

    def learn(self, vectors, labels, *, teacher=False, correct=False):
        if not labels or len(vectors) != len(labels):
            raise PrototypeError("Every training context needs a next-token label.")
        if any(type(label) is not int or label < END or label >= 257 for label in labels):
            raise PrototypeError("Invalid next-token label.")
        normalized = [unit(vector) for vector in vectors]
        if any(vector.shape != (DIMENSIONS,) for vector in normalized):
            raise PrototypeError("Invalid lesson feature dimensions.")
        question = normalized[0][:QUESTION_DIMENSIONS]
        if any(not np.array_equal(vector[:QUESTION_DIMENSIONS], question) for vector in normalized):
            raise PrototypeError("A lesson must contain one question and its answer prefixes.")

        # All mutation is staged. Capacity failure must also roll back corrected
        # memberships, centers, counts, and the new question exemplar.
        samples = self.samples
        sample = self._questions.get(question_key(question))
        if sample is None:
            if len(samples) >= self.max_prototypes:
                raise MemoryFull("Jayce's question feature memory is full. Increase capacity with ./jayce train --capacity N.")
            sample = len(samples)
            samples = np.vstack((samples, question))
        rows = list(self.vectors)
        saved_labels = self.labels.tolist()
        members = [dict(group) for group in self.members]
        prefixes = {key: list(indices) for key, indices in self._prefixes.items()}
        free = []

        def center(group):
            ids, weights = list(group), list(group.values())
            return unit(np.average(samples[ids], axis=0, weights=weights)) / SCALE

        if correct:
            targets = {prefix_key(vector): label for vector, label in zip(normalized, labels)}
            for i, group in enumerate(members):
                key = prefix_key(rows[i])
                if sample in group and targets.get(key) != saved_labels[i]:
                    del group[sample]
                    if group:
                        rows[i] = np.concatenate((center(group), rows[i][QUESTION_DIMENSIONS:]))
                    else:
                        prefixes[key].remove(i)
                        free.append(i)

        for vector, label in zip(normalized, labels):
            key = prefix_key(vector)
            indices = prefixes.setdefault(key, [])
            nearest = next((i for i in indices if saved_labels[i] == label and sample in members[i]), None)
            if nearest is None:
                candidates = sorted(
                    (i for i in indices if saved_labels[i] == label),
                    key=lambda i: float(rows[i][:QUESTION_DIMENSIONS] @ question), reverse=True,
                )
                rivals = [i for i in indices if saved_labels[i] != label]
                for i in candidates:
                    if float(2 * rows[i][:QUESTION_DIMENSIONS] @ question) < JOIN_SIMILARITY:
                        break
                    proposed = dict(members[i])
                    proposed[sample] = 1
                    moved = center(proposed)
                    support = samples[list(proposed)]
                    # Limit the entire cluster, not just the latest input. This
                    # prevents a chain of weak similarities from drifting away.
                    if np.min(2 * (support @ moved)) < MIN_SIMILARITY:
                        continue
                    if rivals:
                        rival_vectors = np.stack([rows[j][:QUESTION_DIMENSIONS] for j in rivals])
                        if np.any(2 * (support @ moved) - (2 * support @ rival_vectors.T).max(axis=1) < MIN_MARGIN):
                            continue
                        # Moving this center must not swallow a rival's examples.
                        if any(np.any(2 * samples[list(members[j])] @ (rows[j][:QUESTION_DIMENSIONS] - moved) < MIN_MARGIN)
                               for j in rivals):
                            continue
                    nearest = i
                    break

            if nearest is None:
                if free:
                    nearest = free.pop()
                    rows[nearest], saved_labels[nearest], members[nearest] = vector, label, {}
                else:
                    if len(rows) >= self.max_prototypes:
                        raise MemoryFull("Jayce's prototype memory is full. This example was not saved; "
                                         "increase capacity with ./jayce train --capacity N.")
                    nearest = len(rows)
                    rows.append(vector)
                    saved_labels.append(label)
                    members.append({})
                indices.append(nearest)
            members[nearest][sample] = members[nearest].get(sample, 0) + 1
            rows[nearest] = np.concatenate((center(members[nearest]), vector[QUESTION_DIMENSIONS:]))

        keep = [i for i, group in enumerate(members) if group]
        saved_vectors = np.stack([rows[i] for i in keep])
        saved_labels = np.asarray([saved_labels[i] for i in keep], dtype=np.int32)
        saved_members = [members[i] for i in keep]
        saved_counts = np.asarray([sum(group.values()) for group in saved_members], dtype=np.int64)
        self.samples, self.vectors, self.labels = samples, saved_vectors, saved_labels
        self.members, self.counts = saved_members, saved_counts
        self.examples += 1
        self.teacher_examples += int(teacher)
        self._index()

    def match(self, vector, min_similarity=MIN_SIMILARITY, min_margin=MIN_MARGIN):
        if not len(self):
            return None
        query = unit(vector)
        if query.shape != (DIMENSIONS,):
            raise PrototypeError("Invalid lesson feature dimensions.")
        indices = self._prefixes.get(prefix_key(query), [])
        if not indices:
            return None
        sample = self._questions.get(question_key(query))
        exact = {int(self.labels[i]) for i in indices if sample in self.members[i]}
        if exact:
            return Match(min(exact), 1.0, len(exact) == 1)
        # Prefix/number/negation guards already matched. Compare only question
        # features, so a long shared answer prefix cannot hide a different topic.
        similarities = np.clip(2 * self.vectors[indices, :QUESTION_DIMENSIONS] @ query[:QUESTION_DIMENSIONS], -1, 1)
        winner = int(np.argmax(similarities))
        token = int(self.labels[indices[winner]])
        similarity = float(similarities[winner])
        rivals = similarities[self.labels[indices] != token]
        margin = similarity - float(rivals.max()) if rivals.size else 2.0
        return Match(token, similarity, similarity >= min_similarity and margin >= min_margin)

    def checkpoint_arrays(self):
        assignments = [(i, sample, count) for i, group in enumerate(self.members) for sample, count in group.items()]
        return {**super().checkpoint_arrays(), "samples": self.samples,
                "members": np.asarray(assignments, dtype=np.int64).reshape(-1, 3)}

    def restore_arrays(self, data):
        super().restore_arrays(data)
        self.samples = np.asarray(data["samples"], dtype=np.float32)
        assignments = data["members"]
        if assignments.ndim != 2 or assignments.shape[1] != 3 or assignments.dtype.kind not in "iu":
            raise PrototypeError("Invalid prototype memberships.")
        self.members = [{} for _ in self.labels]
        for prototype, sample, count in assignments:
            if (not 0 <= prototype < len(self) or not 0 <= sample < len(self.samples) or count < 1
                    or int(sample) in self.members[prototype]):
                raise PrototypeError("Invalid prototype membership.")
            self.members[prototype][int(sample)] = int(count)

    def _validate_saved_arrays(self, vocab_size):
        super()._validate_saved_arrays(vocab_size)
        if (self.samples.ndim != 2 or self.samples.shape[1] != QUESTION_DIMENSIONS
                or not np.isfinite(self.samples).all()
                or not np.allclose(np.linalg.norm(self.samples, axis=1), 1 / SCALE, atol=1e-4)
                or (len(self) and self.vectors.shape[1] != DIMENSIONS)
                or len(self.samples) > self.max_prototypes
                or len(self.members) != len(self)):
            raise PrototypeError("Invalid saved question features.")
        if len(self) and not np.allclose(np.abs(self.vectors[:, QUESTION_DIMENSIONS:]), 1 / np.sqrt(2 * PREFIX_DIMENSIONS), atol=1e-6):
            raise PrototypeError("Invalid saved answer prefix features.")
        for i, group in enumerate(self.members):
            if not group or sum(group.values()) != self.counts[i]:
                raise PrototypeError("Invalid prototype membership counts.")
            center = unit(np.average(self.samples[list(group)], axis=0, weights=list(group.values()))) / SCALE
            if not np.allclose(center, self.vectors[i, :QUESTION_DIMENSIONS], atol=1e-5):
                raise PrototypeError("Prototype center differs from its examples.")
        self._index()
        if len(self._questions) != len(self.samples):
            raise PrototypeError("Duplicate saved question features.")
