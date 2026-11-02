# Robot-base multi-object detections

Use test_picking's Body depth, inward surface normal and target transform unchanged.
Publish latest frame timestamp, sequence, count and at most eight score/bbox/TF
records in detection_shm. Sort valid poses by confidence. Empty detections and
invalid flange/calibration publish zero objects. Keep target_shm unchanged.

1. Add common/shm_detection.py: fixed NumPy layout, locked snapshots, owner cleanup through main.py/ProductIpc only.
2. Add optional explicit calibration arguments to picking geometry helpers and
   call them from detector/xpu_detection.py; connect timestamp-matched FlangeShm.
3. Test 0/1/8/overflow snapshots, shrinking counts, base transforms and rejection
   of missing depth/invalid flange. Run existing picking-related tests.

Hardware validation requires camera and robot; automated checks use synthetic depth.
