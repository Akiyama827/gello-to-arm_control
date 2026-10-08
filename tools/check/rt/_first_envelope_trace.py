"""Opt-in PC/client timestamps for the fake-RT first-envelope diagnosis."""
from __future__ import annotations

import threading
import time

from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtBackend


class TraceBackend(RtBackend):
    def __init__(self, *args, events, run, **kwargs):
        self.events, self.run = events, run
        self._trace_local = threading.local()
        self._first_state = set()
        super().__init__(*args, **kwargs)

    def record(self, event, **fields):
        self.events.append(dict(run=self.run, t_ns=time.monotonic_ns(),
                                event=event, **fields))

    def _reference_exchange(self, operation, **kwargs):
        label = {p.CTL_REFERENCE_COMMIT: "commit",
                 p.CTL_CANCEL_REFERENCE: "cancel"}.get(operation)
        if label:
            self.record(label + "_call")
            self._trace_local.operation = label
        try:
            result = super()._reference_exchange(operation, **kwargs)
            if label:
                self.record(label + "_reply", result=result.get("result"))
            return result
        except Exception as exc:
            if label:
                self.record(label + "_error", error=repr(exc))
            raise
        finally:
            if label:
                self._trace_local.operation = None

    def _send_frame(self, pkt, *, sock=None):
        label = getattr(self._trace_local, "operation", None)
        if label:
            self.record(label + "_send_start")
        result = super()._send_frame(pkt, sock=sock)
        if label:
            self.record(label + "_send_done")
        return result

    def latest_state(self):
        state, received = super().latest_state()
        ref = None if state is None else state.reference
        if ref and ref.get("reference_id") and ref.get("status") in (
                p.REFERENCE_RUNNING, p.REFERENCE_CANCELLED):
            status = int(ref["status"])
            if status not in self._first_state:
                self._first_state.add(status)
                self.record("first_state", status=status,
                            reference_id=ref["reference_id"],
                            received_ns=int(received * 1e9))
        return state, received

    def apply_reference_envelope(self, command, **kwargs):
        self.record("envelope_attempt", reference_id=kwargs["reference_id"],
                    sample_received_ns=int(kwargs["sample_received_at"] * 1e9))
        result = super().apply_reference_envelope(command, **kwargs)
        self.record("envelope_result", seq=result)
        return result
