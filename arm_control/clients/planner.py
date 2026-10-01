"""A reviewed-trajectory client: submit, release, wait. Admission and the review belong to the caller and the supervisor."""
from arm_control.clients.common import Report
from arm_control.execution.contract import Phase, Start


def run_reviewed(arm, request, *, timeout_s: float) -> tuple[Report, object]:
    """Submit ``request`` (a TrajectoryRequest whose review identity the caller already established), release it if
    it starts on release, and wait up to ``timeout_s``. Returns the report and the final ExecutionStatus. A timeout leaves the
    execution running (the SDK never cancels on wait); the caller decides to cancel."""
    report = Report()
    status = arm.execute(request)
    if status.phase is Phase.REJECTED:
        report.refused.append((request.request_id, status.reason))
        return report, status
    report.requested.append(request.request_id)
    report.admitted.append(request.request_id)  # execute() returns after admission: this reply IS the admission
    if request.start is Start.ON_RELEASE and status.phase is Phase.ADMITTED:
        status = arm.release(status.execution_id)
    try:
        status = arm.wait(status.execution_id, timeout_s)
    except TimeoutError as exc:
        report.notes.append(str(exc))
        status = arm.status(status.execution_id)
    return report, status
