"""Coalesce display updates without dropping session completion or popup actions."""

import queue


class SessionEventQueue(queue.Queue):
    """Raw samples and predictions are persisted before entering this UI mailbox.

    Only the newest display value is needed while Tk is busy. Errors, completion,
    and explicit popup transitions remain ordered and lossless.
    """

    @staticmethod
    def _key(item):
        kind, payload = item
        if kind in {"frame", "prediction", "empty", "warming", "input_status"}:
            return kind
        if kind == "guidance" and payload.popup_action is None:
            return kind
        if kind == "guidance_status":
            return kind, payload.get("state")
        return None

    def _put(self, item):
        key = self._key(item)
        if key is not None:
            for previous in self.queue:
                if self._key(previous) == key:
                    self.queue.remove(previous)
                    self.unfinished_tasks -= 1
                    break
        super()._put(item)
