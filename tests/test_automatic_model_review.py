"""Review controls must reflect persisted automatic validation results."""
from types import SimpleNamespace
from unittest import TestCase, mock

from software.posture_app import PostureApplication


class AutomaticReviewTests(TestCase):
    def test_approval_requires_both_checks_and_pending_active_candidate(self):
        model = SimpleNamespace(approval_status="pending", revocation_status="active",
                                validation_status="passed", deployment_validation_status="passed",
                                runtime_compatible=True, sensor_compatible=True)
        app = SimpleNamespace(managed_model_tree=mock.Mock(), _managed_models={"candidate": model},
                              managed_approve_button=mock.Mock(), managed_reject_button=mock.Mock())
        app.managed_model_tree.selection.return_value = ("candidate",)
        PostureApplication.update_managed_review_actions(app)
        app.managed_approve_button.configure.assert_called_with(state="normal")
        for changes in ({"validation_status": "failed"}, {"deployment_validation_status": "unvalidated"},
                        {"runtime_compatible": False}, {"approval_status": "rejected"},
                        {"revocation_status": "revoked"}):
            with mock.patch.multiple(model, **changes):
                PostureApplication.update_managed_review_actions(app)
                app.managed_approve_button.configure.assert_called_with(state="disabled")
        app.managed_model_tree.selection.return_value = ()
        with mock.patch("software.posture_app.messagebox.showerror") as popup:
            PostureApplication.update_managed_review_actions(app)
            popup.assert_not_called()
        app.managed_reject_button.configure.assert_called_with(state="disabled")

    def test_review_clicks_do_not_request_extra_confirmation(self):
        model = object()
        app = SimpleNamespace(selected_managed_model=lambda: model, registry=mock.Mock(),
                              show_management=mock.Mock())
        with mock.patch("software.posture_app.messagebox.askyesno") as confirm, mock.patch(
            "software.posture_app.simpledialog.askstring"
        ) as reason:
            PostureApplication.approve_managed_model(app)
            PostureApplication.reject_managed_model(app)
            confirm.assert_not_called()
            reason.assert_not_called()
        app.registry.approve.assert_called_once()
        app.registry.reject.assert_called_once_with(model, reason="사용자가 모델 관리에서 거절함")
