"""Bulk review preserves validation gates, model identities, and partial results."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

from src.model_management import eligible, identity, review_models, evaluation_summary, EvaluationSummary, ordered_models
from src.model_runtime import ModelRegistry
from test_model_lifecycle import write_v2_candidate
from software.model_management_view import ModelManagementView


class AutomaticReviewTests(TestCase):
    def test_approval_requires_both_checks_and_pending_active_candidate(self):
        model = SimpleNamespace(approval_status="pending", revocation_status="active",
                                validation_status="passed", deployment_validation_status="passed",
                                runtime_compatible=True, sensor_compatible=True)
        self.assertTrue(eligible(model, "approve"))
        for changes in ({"validation_status": "failed"}, {"deployment_validation_status": "unvalidated"},
                        {"runtime_compatible": False}, {"sensor_compatible": False},
                        {"approval_status": "rejected"}, {"revocation_status": "revoked"}):
            with mock.patch.multiple(model, **changes):
                self.assertFalse(eligible(model, "approve"))
        self.assertTrue(eligible(model, "reject"))
        self.assertTrue(eligible(model, "revoke"))

    def test_empty_or_ineligible_selection_does_not_request_confirmation(self):
        view = SimpleNamespace(busy=False, selected=lambda: [])
        with mock.patch("software.model_management_view.simpledialog.askstring") as reason:
            for action in ("approve", "reject", "revoke"):
                ModelManagementView.run_action(view, action)
            reason.assert_not_called()

    def candidate(self, root):
        write_v2_candidate(root)
        registry = ModelRegistry(root, root)
        model, = registry.management_models()
        return registry, model

    def test_batch_uses_real_lifecycle_does_not_activate_and_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, model = self.candidate(root)
            model = registry.validate_deployment(registry.validate_runtime(model))
            before = model.manifest_path.read_bytes(), model.artifact_path.read_bytes()
            progress = []
            result = review_models(registry, [model, model], "approve", actor="test-reviewer", progress=lambda *args: progress.append(args))
            self.assertEqual(len(result.succeeded), 1)
            self.assertEqual(result.succeeded[0].approval_status, "approved")
            self.assertEqual(progress, [(1, 1)])
            self.assertFalse(registry.active_path.exists())
            result = review_models(registry, result.succeeded, "revoke", actor="test-reviewer", reason="batch retirement")
            self.assertEqual(len(result.succeeded), 1)
            self.assertEqual(registry.management_models()[0].revocation_status, "revoked")
            self.assertEqual(registry.store.get(model).reason, "batch retirement")
            self.assertEqual((model.manifest_path.read_bytes(), model.artifact_path.read_bytes()), before)
            self.assertEqual(registry.discover(), [])

    def test_stale_selection_cannot_approve_rejected_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry, model = self.candidate(Path(temporary))
            model = registry.validate_deployment(registry.validate_runtime(model))
            registry.reject(model, reason="another review")
            result = review_models(registry, [model], "approve", actor="test")
            self.assertEqual(len(result.skipped), 1)
            self.assertFalse(result.succeeded)
            self.assertEqual(registry.store.get(model).approval_status, "rejected")

    def test_mixed_batch_continues_after_failure_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, model = self.candidate(Path(temporary))
            ready = replace(model, validation_status="passed", deployment_validation_status="passed", runtime_compatible=True, sensor_compatible=True)
            second = replace(ready, artifact_id="second")
            failed = replace(ready, artifact_id="failed", validation_status="failed")
            changed = replace(ready, artifact_id="changed", manifest_hash="changed")
            registry = mock.Mock()
            registry.management_models.return_value = [ready, second, failed]
            registry.approve.side_effect = [OSError("read only"), replace(second, approval_status="approved")]
            result = review_models(registry, [ready, ready, failed, second, changed], "approve", actor="test")
            self.assertEqual(len(result.succeeded), 1)
            self.assertEqual(len(result.failed), 2)
            self.assertEqual(len(result.skipped), 1)
            self.assertEqual(registry.approve.call_count, 2)
            registry.activate.assert_not_called()

    def test_real_evidence_missing_metrics_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, model = self.candidate(Path(temporary))
            # The fixture only stores F1 and confusion matrix; accuracy is derived correctly.
            summary = evaluation_summary(model)
            self.assertEqual(summary.accuracy, 1.0)
            self.assertEqual(summary.macro_f1, 1.0)
            self.assertEqual(summary.source, "최종 평가")
            self.assertIsNone(evaluation_summary(replace(model, metadata={})).accuracy)
            (Path(temporary) / "artifacts/evaluation.json").write_text('{}')
            summary = evaluation_summary(model)
            self.assertIsNone(summary.accuracy)
            self.assertEqual(summary.source, "근거 확인 필요")

    def test_sort_keeps_missing_values_last_and_preserves_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, model = self.candidate(Path(temporary))
            models = [replace(model, artifact_id=name) for name in ("best", "zero", "missing")]
            summaries = {identity(item): EvaluationSummary(accuracy=score) for item, score in zip(models, (0.9, 0.0, None))}
            self.assertEqual([item.artifact_id for item in ordered_models(models, summaries, "accuracy", True)], ["best", "zero", "missing"])
            self.assertEqual([item.artifact_id for item in ordered_models(models, summaries, "accuracy", False)], ["zero", "best", "missing"])
            self.assertEqual(ModelManagementView.percent(0), "0.00%")
            self.assertEqual(ModelManagementView.percent(None), "미기록")
