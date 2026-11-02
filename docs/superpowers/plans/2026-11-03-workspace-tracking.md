# Workspace-based retention

User confirmed the robot-base origin is the center of a 1.4 m sphere.
Confirmed tracks persist inside that sphere without a last-seen timeout; new
unconfirmed candidates retain the existing one-second timeout. Use predicted
base XYZ for removal; measurements may correct a boundary prediction before removal.
Objects outside the sphere cannot start new tracks. The sphere is an ID-retention
policy, not an inverse-kinematics reachability check.

Tests cover long absence/reacquisition, spherical bounds, candidate expiry,
retained predictions during detector loss, capture/output epoch separation,
workspace-bounded servo predictions and independent tracker-state freshness.
Reuse the existing SHM record (last_seen, observed, covariance). Publish coasting
snapshots at 20 Hz during input loss; use monotonic time only for scheduling.
Expose AGE in the dashboard. No robot commands or SHM ownership changes.
