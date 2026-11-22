"""UI failures must remain recoverable without changing data or starting work."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from software import posture_app as module
from src.app_services import TrainingWizardState


class UIWorkflowTests(TestCase):
    def test_failed_status_save_does_not_enter_training(self):
        app = SimpleNamespace(set_dataset_status=mock.Mock(return_value=False),
                              start_training_wizard=mock.Mock())
        module.PostureApplication.queue_and_train(app, Path('/missing/manifest.json'))
        app.start_training_wizard.assert_not_called()

    def test_missing_csv_does_not_create_recording(self):
        app = SimpleNamespace(mode=mock.Mock(), selected_model=object(), _worker=None,
                              window_choice=mock.Mock(), input_path=mock.Mock())
        app.mode.get.return_value = 'measurement'
        app.window_choice.get.return_value = '3'
        app.input_path.get.return_value = '/does-not-exist/posture-input.csv'
        with mock.patch.object(module, 'MeasurementRecorder') as recorder, mock.patch.object(module.messagebox, 'showerror') as error:
            module.PostureApplication.start_measurement(app)
        recorder.assert_not_called()
        error.assert_called_once()

    def test_hidden_ineligible_preselection_is_removed(self):
        good = SimpleNamespace(csv_path=Path('/good.csv'))
        bad = SimpleNamespace(csv_path=Path('/bad.csv'))
        service = SimpleNamespace(is_training_eligible=lambda r: r is good)
        app = SimpleNamespace(dataset_service=service, show_training_wizard_setup=mock.Mock())
        module.PostureApplication.start_training_wizard(app, [good, bad])
        self.assertEqual(app.selected_records, [good])
        self.assertEqual(app.training_wizard_state.selected_record_paths, (good.csv_path,))

    def test_seed_is_rejected_before_leaving_setup(self):
        from types import MethodType
        from software.experiment_wizard import ExperimentWizardMixin
        for seed in ('-1', '4294967296', 'x'):
            app = SimpleNamespace(
                training_wizard_state=TrainingWizardState(),
                wizard_seed=mock.Mock(), wizard_name=mock.Mock(), wizard_error=mock.Mock(),
                wizard_model_vars=None, personal_retrospective=mock.Mock(),
                show_training_wizard_dataset=mock.Mock())
            app.wizard_seed.get.return_value = seed
            app.wizard_name.get.return_value = '시드 검증'
            app.personal_retrospective.get.return_value = False
            app._remember_experiment_setup = MethodType(ExperimentWizardMixin._remember_experiment_setup, app)
            module.PostureApplication._wizard_to_dataset(app)
            app.show_training_wizard_dataset.assert_not_called()
            app.wizard_error.set.assert_called_once()

    def test_personal_training_exit_without_general_model_returns_home(self):
        app = SimpleNamespace(selected_model=None, show_model_select=mock.Mock())
        module.PostureApplication.show_mode_select(app)
        app.show_model_select.assert_called_once()

    def test_model_write_errors_are_shown_without_callback_crash(self):
        for name, operation in (('approve_managed_model', 'approve'), ('reject_managed_model', 'reject'),
                                ('revoke_managed_model', 'revoke'), ('validate_managed_deployment', 'validate_deployment')):
            registry = mock.Mock()
            getattr(registry, operation).side_effect = PermissionError('read only')
            app = SimpleNamespace(selected_managed_model=lambda: object(), registry=registry,
                                  show_management=mock.Mock())
            with mock.patch.object(module.messagebox, 'showerror') as error, mock.patch.object(module.simpledialog, 'askstring', return_value='test'):
                getattr(module.PostureApplication, name)(app)
            error.assert_called_once()
            app.show_management.assert_not_called()

    def test_partial_status_failure_preserves_unwritten_selection(self):
        first, second = [SimpleNamespace(csv_path=Path('/' + name)) for name in ('first.csv', 'second.csv')]
        service = mock.Mock(); service.set_status.side_effect = [None, PermissionError('read only')]
        app = SimpleNamespace(selected_records=[first, second], dataset_service=service,
                              refresh_dataset_table=mock.Mock())
        with mock.patch.object(module.messagebox, 'showerror') as error:
            module.PostureApplication.set_selected_status(app, '학습에서 제외')
        self.assertEqual(app.selected_records, [second])
        error.assert_called_once()

    def test_training_poll_stops_during_window_close(self):
        app = SimpleNamespace(_closing=True, training_job=mock.Mock())
        module.PostureApplication.poll_training(app)
        app.training_job.events.get_nowait.assert_not_called()

    def test_status_change_requires_confirmation(self):
        app = SimpleNamespace(selected_records=[object(), object()], set_selected_status=mock.Mock(return_value=0))
        with mock.patch.object(module.messagebox, 'askyesno', return_value=False) as ask, \
             mock.patch.object(module.messagebox, 'showinfo') as info:
            module.PostureApplication.confirm_selected_status(app, '폐기')
        self.assertIn('2개', ask.call_args.args[1]); self.assertIn('삭제되지 않습니다', ask.call_args.args[1])
        app.set_selected_status.assert_not_called(); info.assert_not_called()
        with mock.patch.object(module.messagebox, 'askyesno', return_value=True), \
             mock.patch.object(module.messagebox, 'showinfo') as info:
            module.PostureApplication.confirm_selected_status(app, '학습에 포함')
        app.set_selected_status.assert_called_once_with('학습에 포함'); info.assert_called_once()

    def test_status_change_without_selection_asks_nothing(self):
        app = SimpleNamespace(selected_records=[], set_selected_status=mock.Mock())
        with mock.patch.object(module.messagebox, 'askyesno') as ask, mock.patch.object(module.messagebox, 'showerror') as error:
            module.PostureApplication.confirm_selected_status(app, '폐기')
        ask.assert_not_called(); error.assert_called_once(); app.set_selected_status.assert_not_called()

    def test_sort_toggles_direction_and_reset_clears_filters(self):
        app = SimpleNamespace(dataset_sort=(None, False), refresh_dataset_table=mock.Mock())
        module.PostureApplication.sort_dataset_table(app, '행')
        self.assertEqual(app.dataset_sort, ('행', False))
        module.PostureApplication.sort_dataset_table(app, '행')
        self.assertEqual(app.dataset_sort, ('행', True))
        module.PostureApplication.sort_dataset_table(app, '실명')
        self.assertEqual(app.dataset_sort, ('실명', False))
        variables = {name: mock.Mock() for name in ('filter_subject', 'filter_posture', 'filter_date',
                                                     'filter_min_rows', 'filter_min_duration', 'filter_status')}
        reset = SimpleNamespace(**variables, refresh_dataset_table=mock.Mock())
        module.PostureApplication.reset_dataset_filters(reset)
        variables['filter_status'].set.assert_called_once_with('전체')
        variables['filter_subject'].set.assert_called_once_with('')
        reset.refresh_dataset_table.assert_called_once()

    def test_source_summary_distinguishes_live_and_replay(self):
        summary = mock.Mock(); path = mock.Mock()
        app = SimpleNamespace(source_summary=summary, input_path=path)
        path.get.return_value = ''
        module.PostureApplication._update_source_summary(app)
        summary.set.assert_called_with('입력: 실시간 센서')
        path.get.return_value = __file__
        module.PostureApplication._update_source_summary(app)
        self.assertIn('저장 파일 재생', summary.set.call_args.args[0])
        path.get.return_value = '/does-not-exist.csv'
        module.PostureApplication._update_source_summary(app)
        self.assertIn('찾을 수 없습니다', summary.set.call_args.args[0])
