# Main IPC Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `main.py` supervisor that exclusively owns the product's three shared-memory segments and camera semaphore while workers only attach to them.

**Architecture:** `common/ipc.py` will encapsulate creation and ordered cleanup of the four IPC resources. The root `main.py` holds that object until SIGINT or SIGTERM, while `camera_runner.py` becomes an attach-only writer. Existing SHM classes remain the source of truth for buffer layouts.

**Tech Stack:** Python 3.12, `multiprocessing.shared_memory`, `posix_ipc`, `fcntl`, `unittest`, NumPy

**Spec:** `docs/superpowers/specs/2026-10-16-main-ipc-ownership-design.md`

## Global Constraints

- Only `main.py` may create, replace, or unlink product IPC resources.
- Workers use `is_owner=False`, call `close()`, and never call `unlink()`.
- This increment does not start, monitor, or restart worker processes.
- Cleanup targets only the exact configured SHM and semaphore names.
- A second supervisor must fail without modifying the first supervisor's resources.

## Review Focus

- A second supervisor using the same lock must fail and leave the first one's SHM readable; test in Task 2.
- A partially failed startup must remove resources created during that attempt; test in Task 2.
- An attached worker closing must not unlink its SHM; test in Task 1.
- SIGTERM must take the same cleanup path as SIGINT; test the shared stop event in Task 4.
- Missing IPC must make `camera_runner` return a clear error rather than becoming an owner; test in Task 3.

---

### Task 1: Consistent Shared-Memory Ownership

**Files:**
- Modify: `common/shm_camera.py:49-68,170-182`
- Modify: `common/shm_flange.py:65-85,193-199`
- Modify: `common/shm_target.py:73-88,156-168`
- Create: `tests/test_shm_ownership.py`

**Interfaces:**
- Consumes: Existing constructors `CameraShm(..., is_owner: bool)`, `FlangeShm(..., is_owner: bool)`, and `TargetShm(..., is_owner: bool)`.
- Produces: A uniform contract where owners create a new zeroed segment and unlink it on `close()`, while attached instances only close their handles.

- [ ] **Step 1: Write failing ownership tests**

Add `unittest` cases with unique names asserting owner-created headers are zero, an attached instance can exchange representative data, attached `close()` leaves the segment attachable, and owner `close()` removes it. Include all three SHM classes.

- [ ] **Step 2: Run tests and verify the ownership mismatch**

Run: `.venv/bin/python -m unittest -v tests.test_shm_ownership`

Expected: FAIL because `FlangeShm` owner close does not unlink and owner creation behavior is inconsistent.

- [ ] **Step 3: Implement the uniform owner contract**

For each class, make `is_owner=True` call `SharedMemory(name=..., create=True, size=...)`, zero the complete buffer, and unlink on owner `close()`. Do not silently attach to or delete an existing segment in these constructors. Keep `is_owner=False` unchanged.

- [ ] **Step 4: Verify Task 1**

Run: `.venv/bin/python -m unittest -v tests.test_shm_ownership`

Expected: all Task 1 tests PASS with no resource-tracker warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add common/shm_camera.py common/shm_flange.py common/shm_target.py tests/test_shm_ownership.py
git commit -m "refactor: standardize shared memory ownership"
```

### Task 2: Product IPC Lifecycle

**Files:**
- Create: `common/ipc.py`
- Create: `tests/test_ipc.py`

**Interfaces:**
- Consumes: Uniform owner constructors from Task 1.
- Produces: `ProductIpc(camera_shm_name: str = "camera_shm", flange_shm_name: str = "shm_flange", target_shm_name: str = "shm_target", camera_sem_name: str = "/camera_sem", lock_path: str = "/tmp/seongju_melon_detector.lock")`; methods `start() -> None` and `close() -> None`; context-manager support.

- [ ] **Step 1: Write failing lifecycle tests**

Add `unittest` cases asserting `start()` creates all four resources with false statuses, consumers can attach, consumer close preserves resources, `close()` removes all resources, duplicate lock acquisition raises `RuntimeError` without changing live data, and injected failure during startup cleans only resources created by that instance.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m unittest -v tests.test_ipc`

Expected: ERROR because `common.ipc.ProductIpc` does not exist.

- [ ] **Step 3: Implement `ProductIpc`**

Use non-blocking `fcntl.flock` on the exact lock path before stale cleanup. After locking, unlink stale resources by their exact names, create the three owner objects and semaphore, and set statuses false. Track successfully created resources and clean them in reverse order on failure or `close()`; set all available statuses false before cleanup and make `close()` idempotent.

- [ ] **Step 4: Verify Task 2 and ownership regression**

Run: `.venv/bin/python -m unittest -v tests.test_ipc tests.test_shm_ownership`

Expected: all tests PASS with no leaked IPC resources.

- [ ] **Step 5: Commit Task 2**

```bash
git add common/ipc.py tests/test_ipc.py
git commit -m "feat: add product IPC lifecycle manager"
```

### Task 3: Attach-Only Camera Worker

**Files:**
- Modify: `camera/camera_runner.py:22-58,103-120`
- Create: `tests/test_camera_runner_ipc.py`

**Interfaces:**
- Consumes: IPC names and resources created by `ProductIpc`.
- Produces: `open_camera_ipc() -> tuple[CameraShm, posix_ipc.Semaphore]`, which only attaches and raises `RuntimeError("main.py를 먼저 실행하세요")` when either resource is missing.

- [ ] **Step 1: Write failing attach-only tests**

Test that `open_camera_ipc()` fails when resources are absent, attaches when `ProductIpc` is active, and closing returned handles does not unlink either resource.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m unittest -v tests.test_camera_runner_ipc`

Expected: ERROR because `open_camera_ipc()` does not exist.

- [ ] **Step 3: Convert `camera_runner.py` to attach-only**

Move the attach logic into `open_camera_ipc()`. Construct `CameraShm(..., is_owner=False)` and open the existing semaphore without `O_CREAT`. Remove old-resource deletion, SHM creation, semaphore unlink, and unused `sys` import. Keep writer status handling and only close handles during finalization.

- [ ] **Step 4: Verify Task 3 and IPC regressions**

Run: `.venv/bin/python -m unittest -v tests.test_camera_runner_ipc tests.test_ipc tests.test_shm_ownership`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add camera/camera_runner.py tests/test_camera_runner_ipc.py
git commit -m "refactor: make camera worker attach only"
```

### Task 4: Root IPC Supervisor

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `ProductIpc` from Task 2.
- Produces: `install_signal_handlers(stop_event: threading.Event) -> None`, `run(stop_event: threading.Event | None = None) -> int`, and CLI `main() -> int`.

- [ ] **Step 1: Write failing supervisor tests**

Test that `run()` starts `ProductIpc`, waits until the supplied event is set, then closes it; test handlers installed by `install_signal_handlers()` set that same event for both SIGINT and SIGTERM without embedding cleanup in the handler.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m unittest -v tests.test_main`

Expected: ERROR because root `main.py` does not exist.

- [ ] **Step 3: Implement the supervisor entry point**

Use a `threading.Event` as the stop signal. Register SIGINT and SIGTERM handlers in `main()`, enter `ProductIpc`, print readiness, wait on the event, and return `0` after context cleanup. Do not launch workers in this increment.

- [ ] **Step 4: Run complete verification**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all IPC tests PASS without warnings.

Run: `.venv/bin/python -m py_compile main.py common/ipc.py common/shm_camera.py common/shm_flange.py common/shm_target.py camera/camera_runner.py`

Expected: exit code 0.

- [ ] **Step 5: Commit Task 4**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add main IPC supervisor"
```
