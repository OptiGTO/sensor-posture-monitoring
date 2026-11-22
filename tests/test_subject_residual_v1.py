"""Synthetic tests only: subject geometry, scope, deterministic inference and audit."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src import subject_residual_v1 as subject
from src.posture_core import feature_names


def fixture(offsets=None, dtype=np.float32):
    if offsets is None:
        offsets = np.asarray([[3, 2, 1], [3, -2, -1], [-3, 2, -1], [-3, -2, 1]], dtype=float)
    X, y, groups, records = [], [], [], []
    for person, offset in enumerate(offsets):
        for label in range(7):
            for record in range(2):
                row = np.zeros(596, dtype=float)
                row[:len(offset)] = offset
                row[0] += 5.  # TRAIN centroid mean must not be subtracted at inference.
                row[10], row[11] = label - 3., (label - 3.) ** 2
                row[12] = .25 if record else -.25
                count = 1 + (person + label + record) % 3
                for jitter in np.linspace(-.1, .1, count) if count > 1 else [0.]:
                    observation = row.copy()
                    observation[13] = jitter
                    X.append(observation)
                    y.append(label)
                    groups.append(f"person-{person}")
                    records.append(f"person-{person}/class-{label}/record-{record}")
    return np.asarray(X, dtype=dtype), np.asarray(y, dtype=np.int64), np.asarray(groups), np.asarray(records)


class SubjectResidualTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.records = fixture()

    def fit(self, mode="remove_subject3", X=None, y=None, groups=None, records=None):
        return subject.SubjectResidualV1(mode).fit(self.X if X is None else X,
            self.y if y is None else y, groups=self.groups if groups is None else groups,
            record_ids=self.records if records is None else records)

    def test_identity_returns_original_array_without_arithmetic_or_svd(self):
        for dtype in (np.float32, np.float64):
            X = self.X.astype(dtype)
            X[0, 23] = -0.
            with patch.object(subject.np.linalg, "svd", side_effect=AssertionError("identity SVD")):
                model = self.fit("identity", X=X)
                query = X[::-1]
                before = query.tobytes()
                self.assertIs(model.transform(query), query)
                self.assertEqual(model.transform(query).tobytes(), before)
            self.assertIsNone(model.components_)
            self.assertIsNone(model.record_means_)
            self.assertIsNone(model.training_details_["diagnostics"])
            self.assertTrue(subject.verify_subject_residual(model, "identity"))

    def test_known_subject_offset_removed_and_orthogonal_class_signal_preserved(self):
        model = self.fit()
        output = model.transform(self.X)
        np.testing.assert_allclose(output[:, :3], 0., atol=1e-13)
        np.testing.assert_allclose(output[:, 3:], self.X[:, 3:], rtol=0, atol=1e-13)
        self.assertEqual(output.dtype, np.dtype(np.float64))
        self.assertEqual(output.shape, self.X.shape)
        np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), rtol=0, atol=1e-13)
        self.assertTrue(model.components_.flags.c_contiguous)
        anchors = np.argmax(np.abs(model.components_), axis=1)
        self.assertTrue(np.all(model.components_[np.arange(3), anchors] > 0))
        self.assertEqual(model.rank_tolerance_, np.finfo(float).eps * 596 * model.singular_values_[0])
        details = model.training_details_["diagnostics"]
        self.assertAlmostEqual(details["subject_centroid_energy_fraction_removed"], 1.)
        self.assertAlmostEqual(details["class_centroid_contrast_energy_fraction_removed"], 0.)
        self.assertEqual(len(details["class_centroid_pair_distances"]), 21)
        front_back = next(row for row in details["class_centroid_pair_distances"] if row["classes"] == [1, 2])
        self.assertAlmostEqual(front_back["squared_distance_before"], front_back["squared_distance_after"])
        # Returning a centered residual would preserve the global offset5; the specified z-zP removes it.
        self.assertAlmostEqual(model.centroid_mean_[0], 5.)
        self.assertAlmostEqual(output[:, 0].mean(), 0.)

    def test_record_then_posture_then_person_means_have_equal_contributions(self):
        X = self.X.astype(np.float64)
        selected_record = self.records[0]
        mask = self.records == selected_record
        # Unequal row counts distinguish a record mean from a pooled-window mean.
        X[mask, 22] = 30.
        model = self.fit(X=X)
        metadata = model.record_metadata_
        expected_record = np.asarray([X[row["row_indices"]].mean(0) for row in metadata])
        np.testing.assert_allclose(model.record_means_, expected_record, rtol=0, atol=1e-15)
        chosen = [i for i, row in enumerate(metadata) if row["subject"] == "person-0" and row["class_id"] == 0]
        self.assertAlmostEqual(model.person_class_means_[0, 0, 22], 15.)
        self.assertAlmostEqual(model.person_centroids_[0, 22], 15. / 7)
        self.assertAlmostEqual(model.centroid_mean_[22], 15. / 28)
        self.assertFalse(np.array_equal(model.person_class_means_[0, 0], X[(self.groups == "person-0") & (self.y == 0)].mean(0)))
        # Repeating all rows of just one record must not change learned record geometry.
        repeated = self.fit(X=np.concatenate((X, X[mask], X[mask])), y=np.concatenate((self.y, self.y[mask], self.y[mask])),
            groups=np.concatenate((self.groups, self.groups[mask], self.groups[mask])),
            records=np.concatenate((self.records, self.records[mask], self.records[mask])))
        np.testing.assert_array_equal(model.person_class_means_, repeated.person_class_means_)
        np.testing.assert_array_equal(model.components_, repeated.components_)

    def test_query_never_fits_uses_no_identity_and_has_exact_batch_row_order_outputs(self):
        # Dense learned directions exercise floating-point reduction order; a
        # coordinate-aligned synthetic basis would hide layout-dependent errors.
        rng = np.random.default_rng(613)
        dense = (self.X[:, :14].astype(np.float64) @ rng.normal(size=(14, 596))
            + rng.normal(scale=.03, size=self.X.shape)).astype(np.float32)
        model = self.fit(X=dense)
        query = np.concatenate((dense[:4], np.full_like(dense[:2], 500.))).astype(np.float64)
        before, state, audit = query.tobytes(), model.state_sha256_, deepcopy(model.training_details_)
        with patch.object(subject.np.linalg, "svd", side_effect=AssertionError("query fit")), \
                patch.object(subject, "_hierarchy", side_effect=AssertionError("query aggregates")), \
                patch.object(subject, "_scope", side_effect=AssertionError("query identities")):
            expected = model.transform(query)
            separate = np.concatenate([model.transform(query[i:i+1]) for i in range(len(query))])
            reversed_rows = model.transform(query[::-1])
            fortran = model.transform(np.asfortranarray(query))
        self.assertEqual(expected.tobytes(), separate.tobytes())
        self.assertEqual(expected[::-1].tobytes(), reversed_rows.tobytes())
        self.assertEqual(expected.tobytes(), fortran.tobytes())
        self.assertEqual(query.tobytes(), before)
        self.assertEqual(model.training_details_, audit)
        self.assertEqual(subject.subject_residual_state_sha256(model), state)
        self.assertEqual(expected.tobytes(), self.fit(X=dense).transform(query).tobytes())
        self.assertTrue(subject.verify_subject_residual(model, "remove_subject3"))
        with self.assertRaises(TypeError):
            model.transform(query, groups=np.asarray(["unseen"] * len(query)))

    def test_refit_reload_input_nonmutation_and_training_row_permutation(self):
        before = tuple(value.tobytes() for value in (self.X, self.y, self.groups, self.records))
        model = self.fit()
        refit = self.fit()
        memory = io.BytesIO()
        joblib.dump(model, memory)
        memory.seek(0)
        restored = joblib.load(memory)
        self.assertEqual(model.state_sha256_, refit.state_sha256_)
        self.assertEqual(model.state_sha256_, restored.state_sha256_)
        self.assertEqual(model.transform(self.X).tobytes(), refit.transform(self.X).tobytes())
        self.assertEqual(model.transform(self.X).tobytes(), restored.transform(self.X).tobytes())
        self.assertTrue(subject.verify_subject_residual(restored, "remove_subject3"))
        order = np.random.default_rng(501).permutation(len(self.X))
        shuffled = self.fit(X=self.X[order], y=self.y[order], groups=self.groups[order], records=self.records[order])
        np.testing.assert_array_equal(model.components_, shuffled.components_)
        self.assertEqual(model.transform(self.X).tobytes(), shuffled.transform(self.X).tobytes())
        self.assertNotEqual(model.training_input_sha256_, shuffled.training_input_sha256_)
        self.assertEqual(before, tuple(value.tobytes() for value in (self.X, self.y, self.groups, self.records)))

    def test_scope_rejects_missing_postures_impure_records_unknown_or_empty_ids(self):
        mask = ~((self.groups == "person-0") & (self.y == 0))
        for mode in ("identity", "remove_subject3"):
            with self.assertRaisesRegex(ValueError, "Every training person"):
                self.fit(mode, X=self.X[mask], y=self.y[mask], groups=self.groups[mask], records=self.records[mask])
            mask3 = self.groups != "person-3"
            with self.assertRaisesRegex(ValueError, "four training people"):
                self.fit(mode, X=self.X[mask3], y=self.y[mask3], groups=self.groups[mask3], records=self.records[mask3])
        changed = self.records.copy()
        changed[np.flatnonzero(self.groups == "person-1")[0]] = changed[0]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.fit(records=changed)
        changed = self.records.copy()
        changed[np.flatnonzero(self.y == 1)[0]] = changed[0]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.fit(records=changed)
        for column in ("groups", "records"):
            for bad in (None, " ", 3, float("nan")):
                changed = (self.groups if column == "groups" else self.records).astype(object)
                changed[0] = bad
                with self.subTest(column=column, bad=bad), self.assertRaises(ValueError):
                    self.fit(**{column: changed})
        for bad in (self.y.astype(float), self.y[:-1], self.y[:, None], np.zeros_like(self.y)):
            with self.assertRaises(ValueError):
                self.fit(y=bad)

    def test_normalization_scope_digests_and_names(self):
        groups = np.asarray([" " + value + " " for value in self.groups])
        records = np.asarray([" " + value + " " for value in self.records])
        normalized, original = self.fit(groups=groups, records=records), self.fit()
        self.assertEqual(normalized.state_sha256_, original.state_sha256_)
        self.assertEqual(original.training_input_sha256_, subject._array_digest(self.X))
        self.assertEqual(original.training_labels_sha256_, subject._array_digest(self.y))
        self.assertEqual(original.training_groups_sha256_, subject._json_digest(self.groups.tolist()))
        self.assertEqual(original.training_records_sha256_, subject._json_digest(self.records.tolist()))
        self.assertEqual(original.training_details_["class_counts"], np.bincount(self.y).tolist())
        self.assertEqual(original.training_details_["training_records"], 56)
        names = original.get_feature_names_out(feature_names())
        names[0] = "changed"
        self.assertEqual(original.get_feature_names_out()[0], feature_names()[0])
        named = pd.DataFrame(self.X, columns=feature_names())
        self.assertEqual(original.transform(named).tobytes(), original.transform(self.X).tobytes())
        with self.assertRaises(ValueError):
            original.transform(named.iloc[:, ::-1])

    def test_insufficient_rank_boundary_tie_and_solver_failure_fail_without_fallback(self):
        rank2 = self.X.copy()
        rank2[:, 2] = 0.
        for values in (rank2, np.zeros_like(self.X)):
            with self.assertRaisesRegex(ValueError, "rank is below3"):
                self.fit(X=values)
        offsets = np.concatenate((np.diag([4., 3., 2., 2.]), -np.diag([4., 3., 2., 2.])))
        X, y, groups, records = fixture(offsets=offsets, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "boundary is tied"):
            self.fit(X=X, y=y, groups=groups, records=records)
        with patch.object(subject.np.linalg, "svd", side_effect=np.linalg.LinAlgError("did not converge")), \
                self.assertRaisesRegex(ValueError, "did not converge; no fallback"):
            self.fit()

    def test_zero_class_contrast_diagnostic_is_null_and_not_a_fallback(self):
        X = self.X.copy()
        X[:, 10:12] = 0.
        model = self.fit(X=X)
        diagnostic = model.training_details_["diagnostics"]
        self.assertEqual(diagnostic["class_centroid_contrast_energy"], 0.)
        self.assertIsNone(diagnostic["class_centroid_contrast_energy_fraction_removed"])
        self.assertTrue(all(row["fraction_removed"] is None for row in diagnostic["class_centroid_pair_distances"]))
        self.assertEqual(model.components_.shape, (3, 596))

    def test_full_verifier_rejects_coordinates_aggregation_scope_and_audit_tampering(self):
        model = self.fit()
        for mutate in (
            lambda m: setattr(m, "mode", "identity"),
            lambda m: m.components_.__setitem__((0, 1), .5),
            lambda m: setattr(m, "components_", np.asfortranarray(m.components_)),
            lambda m: m.record_means_.__setitem__((0, 10), 100.),
            lambda m: m.person_class_means_.__setitem__((0, 0, 10), 100.),
            lambda m: m.person_centroids_.__setitem__((0, 0), 100.),
            lambda m: m.singular_values_.__setitem__(0, 100.),
            lambda m: m.signs_.__setitem__(0, 0.),
            lambda m: m.record_metadata_[0]["row_indices"].__setitem__(0, 1),
            lambda m: m.classes_.__setitem__(0, 7),
            lambda m: m.class_counts_.__setitem__(0, 1),
            lambda m: m.training_subjects_.__setitem__(0, "unknown-person"),
            lambda m: setattr(m, "training_input_sha256_", "a" * 64),
            lambda m: setattr(m, "rank_tolerance_", 1.),
            lambda m: m.training_details_.update(external_validation_used=True),
            lambda m: m.rule_.update(removed_rank=2),
            lambda m: setattr(m, "state_sha256_", "a" * 64),
        ):
            changed = deepcopy(model)
            mutate(changed)
            with self.assertRaises(ValueError):
                subject.verify_subject_residual(changed, "remove_subject3")
        # Even a newly hashed audit cannot make duplicate TRAIN row coverage valid.
        changed = deepcopy(model)
        changed.record_metadata_[0]["row_indices"][0] = changed.record_metadata_[1]["row_indices"][0]
        changed.state_sha256_ = subject.subject_residual_state_sha256(changed)
        changed.training_details_ = subject._audit(changed)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            subject.verify_subject_residual(changed, "remove_subject3")

    def test_invalid_numeric_contract_and_mutable_fit_modes_fail(self):
        for X in (self.X[:, :-1], self.X[:0], self.X[0], self.X.astype(int), np.full_like(self.X, np.nan), np.full_like(self.X, np.inf)):
            with self.assertRaises(ValueError):
                self.fit(X=X)
        with self.assertRaises(ValueError):
            self.fit("remove_subject2")
        model = self.fit()
        model.set_params(mode="identity")
        with self.assertRaises(ValueError):
            model.transform(self.X)
        model = self.fit()
        model.components_ = np.asfortranarray(model.components_)
        with self.assertRaisesRegex(ValueError, "C-order"):
            model.transform(self.X)
        with self.assertRaises(TypeError):
            subject.SubjectResidualV1().fit(self.X, self.y, groups=self.groups, record_ids=self.records,
                validation_data=object())


if __name__ == "__main__":
    unittest.main()
