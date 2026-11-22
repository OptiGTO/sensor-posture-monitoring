#!/usr/bin/env python3
"""Exercise model review against isolated real artifacts in a macOS Tk session."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="posture-management-qa-") as temporary:
        root = Path(temporary)
        os.environ["POSTURE_APP_STATE_ROOT"] = str(root / "state")
        from software import posture_app as module
        from src.model_runtime import ModelRegistry
        from src.model_management import identity
        from test_model_lifecycle import write_v2_candidate
        # Three distinct identities with real hash-bound evidence and runtime checks.
        paths = [write_v2_candidate(root / "artifacts" / "models" / str(index), artifact_id=f"gui-{index}") for index in range(3)]
        registry = ModelRegistry(root, root)
        for model in registry.management_models():
            registry.validate_deployment(registry.validate_runtime(model))
        (root / "artifacts" / "manifests").mkdir()
        (root / "artifacts" / "manifests" / "broken.json").write_text("{")
        originals = {path: path.read_bytes() for path in paths}
        app = module.PostureApplication()
        app.registry = registry
        failures = []
        app.report_callback_exception = lambda *error: failures.append(str(error))
        checks = []
        try:
            app.show_management(); app.update()
            view = app.model_management
            assert len(view.rows) == 3 and len(view.tree.get_children()) == 4
            # Numeric metrics are visible, errors inspectable, and no real state is used.
            assert all(view.tree.set(iid, "accuracy") == "100.00%" for iid in view.rows)
            with mock.patch("software.model_management_view.messagebox.showerror") as popup:
                view.tree.selection_set("invalid-0"); view.show_detail(); popup.assert_called_once()
            checks.append("metrics_and_invalid_manifest_details")
            first = next(iter(view.rows))
            view.tree.selection_set(first); app.update()
            selected_identity = identity(view.selected()[0])
            view.sort("accuracy"); app.update()
            assert identity(view.selected()[0]) == selected_identity
            view.sort("macro_f1"); app.update()
            assert identity(view.selected()[0]) == selected_identity
            checks.append("sort_preserves_exact_selection")
            view.query.set("gui-0"); app.update()
            assert len(view.rows) == 1
            view.select_visible(); assert len(view.selected()) == 1
            view.query.set("no-such-model"); app.update()
            assert len(view.selected()) == 0 and not view.tree.get_children()
            assert str(view.buttons["approve"].cget("state")) == "disabled"
            view.reset_filters(); app.update()
            checks.append("search_empty_state_clears_hidden_selection")
            iid = next(iter(view.rows)); view.tree.focus(iid)
            view.toggle_focused(); assert len(view.selected()) == 1
            view.toggle_focused(); assert len(view.selected()) == 0
            box = view.tree.bbox(iid, "check")
            view.click_checkbox(type("Event", (), {"x": box[0]+10, "y": box[1]+10})())
            assert len(view.selected()) == 1
            view.clear_selection()
            checks.append("checkbox_and_keyboard_selection")
            view.status_filter.set("승인 가능"); app.update()
            view.select_ready(); app.update()
            assert len(view.selected()) == 3
            assert str(view.activate_button.cget("state")) == "disabled"
            def finish_batch():
                deadline = time.monotonic() + 30
                while view.busy and time.monotonic() < deadline:
                    app.update(); time.sleep(.01)
                app.update()
                assert not view.busy
                assert app.grab_current() is None
            with mock.patch("software.model_management_view.simpledialog.askstring") as prompt:
                view.run_action("approve"); finish_batch()
                prompt.assert_not_called()
            assert all(model.approval_status == "approved" for model in registry.management_models())
            assert not registry.active_path.exists()
            assert len(view.rows) == 0  # The ready filter remains in place after success.
            checks.append("bulk_approval_without_confirmation_or_activation")
            view.status_filter.set("사용 가능"); app.update()
            view.select_visible(); app.update()
            with mock.patch("software.model_management_view.simpledialog.askstring", return_value=" "):
                view.run_action("revoke")
                assert not view.busy and len(view.rows) == 3
            with mock.patch("software.model_management_view.simpledialog.askstring", return_value="GUI test retirement") as prompt:
                view.run_action("revoke"); finish_batch(); prompt.assert_called_once()
            assert len(view.rows) == 0
            assert all(model.revocation_status == "revoked" for model in registry.management_models())
            app.show_management(); app.update()
            assert not app.model_management.rows
            checks.append("bulk_revocation_reason_cancel_and_persistent_hiding")
            assert all(path.read_bytes() == content for path, content in originals.items())
            for size in ("850x700", "980x760"):
                app.geometry(size); app.update()
                for tab in (0, 1):
                    app.management_notebook.select(tab); app.update()
                    assert app.management_notebook.winfo_ismapped()
                    if tab == 0:
                        def check_buttons(widget):
                            for child in widget.winfo_children():
                                if child.winfo_class() == "TButton":
                                    assert child.winfo_ismapped(), (size, child.cget("text"), "hidden")
                                    assert child.winfo_width() >= child.winfo_reqwidth(), (size, child.cget("text"), "compressed")
                                    assert child.winfo_rootx() + child.winfo_width() <= app.winfo_rootx() + app.winfo_width(), (size, child.cget("text"), "outside window")
                                check_buttons(child)
                        check_buttons(app.model_management)
            checks.append("minimum_and_default_window_tabs")
            assert not failures, failures
        finally:
            app.destroy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"real_tk": True, "checks": checks, "callback_errors": failures,
                                          "isolated_state": True, "manifests_unchanged": True}, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"passed": len(checks), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
