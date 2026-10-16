# Main IPC Ownership Design

## Goal

Make `main.py` the single owner of the product's shared-memory segments and
camera semaphore. Worker processes attach to those resources but never create,
unlink, or replace them. Restarting one worker must not invalidate IPC handles
held by the other workers.

This first increment manages IPC lifetime only. Starting, monitoring, and
restarting worker processes remains out of scope.

## IPC Resources

`main.py` owns these resources for its entire lifetime:

| Resource | Writer | Readers |
| --- | --- | --- |
| `camera_shm` | camera worker | detection and preview workers |
| `shm_flange` | flange worker | detection and picking workers |
| `shm_target` | detection or picking worker | robot-control worker |
| `/camera_sem` | camera worker | camera-frame consumer |

Only one camera-frame consumer may wait on `/camera_sem` reliably because a
POSIX semaphore wakes one waiter per release. Supporting multiple independent
consumers requires a later notification design and is not part of this change.

## Ownership Contract

`main.py` creates every IPC resource before workers start. It initializes all
status fields to `False` and leaves data buffers zeroed.

Workers open resources in attach-only mode:

- They use `create=False` or `is_owner=False`.
- They may update the status associated with the data they produce.
- They call `close()` when stopping.
- They never call `unlink()` or replace an existing resource.

On normal shutdown, `main.py` sets all status fields to `False`, closes its
handles, and unlinks all four IPC resources. Worker shutdown must precede this
cleanup once process supervision is added.

## Startup and Stale Resources

At startup, `main.py` removes resources left behind by an earlier crashed
supervisor and creates fresh resources with the current layouts. Cleanup is
limited to the four exact configured names.

Starting a second `main.py` while one is active must fail instead of deleting
the live supervisor's resources. A dedicated supervisor lock prevents the
stale-resource cleanup path from running concurrently in two supervisors.

Workers started before `main.py` fail clearly because attach-only opens cannot
find the required resource. They do not fall back to becoming owners.

## Code Changes

- Add `main.py` as the IPC supervisor entry point.
- Reuse `CameraShm`, `FlangeShm`, and `TargetShm` for resource allocation and
  layout definitions rather than duplicating buffer sizes.
- Change `camera/camera_runner.py` to attach to `camera_shm` and
  `/camera_sem`; remove its startup cleanup and final unlink operations.
- Keep the existing attach-only consumers unchanged unless tests expose an
  ownership violation.
- Make owner initialization deterministic where an existing SHM class does
  not currently zero or reset its header.

## Failure Behavior

- Missing IPC in a worker produces an error directing the operator to start
  `main.py` first.
- Failure to create any IPC resource causes `main.py` to close and unlink only
  resources it successfully created during that attempt.
- SIGINT and SIGTERM trigger the same ordered cleanup path.
- Unexpected worker death does not delete shared memory; its status can remain
  stale until process supervision is added, so consumers must still use
  sequence/timestamp freshness checks.

## Verification

Automated tests will verify:

1. The supervisor creates all three SHM segments and the semaphore.
2. Workers can attach and exchange representative data.
3. Closing an attached worker leaves resources available to other workers.
4. Supervisor cleanup removes every resource.
5. A second supervisor cannot remove resources owned by the first.

Tests use unique IPC names so they cannot interfere with a running product.
