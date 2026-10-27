# EOG Authentication Experiment Protocol

This stage evaluates one-to-one verification while preserving the existing multiclass identification benchmark. It is an academic, within-session benchmark—not a production security evaluation.

## Primary protocol: known impostors

Each of the 27 participants is treated in turn as the authorized user. The positive examples are that participant's windows; all other participants are impostors. Every identity uses the existing chronological 20 training / 7 validation / 7 test split. A `StandardScaler` and class-balanced logistic regression are fitted on training observations only. EER is linearly interpolated between ROC operating points; the acceptance threshold is the nearest finite empirical validation threshold to that crossing and is frozen before test evaluation.

The test report includes per-user and macro-average ROC-AUC, EER from continuous scores, FAR, FRR, balanced accuracy, precision, recall, PR-AUC, confusion counts, thresholds, and observation counts. Pooled score metrics are secondary because each authorized user is a separate verification task. Conventional accuracy is not a headline result: in the known-impostor test, always rejecting produces 182 correct negatives out of 189 requests, or 96.3% accuracy.

## Feature-confound ablations

The same observations and procedure are evaluated with all six features, channel means only, and standard deviation/range only (which excludes both channel means). The last ablation removes summary-feature means. It is not equivalent to centering or correcting raw electrode signals and cannot determine whether DC offsets are physiological or acquisition artifacts.

A single fixed-seed random-window split is also reported as a secondary temporal-optimism diagnostic. It is not the primary protocol, does not replace chronological holdout, and is not an independent replication because it repartitions the same recordings.

## Secondary protocol: unseen impostor identities

For each authorized user and repeat, the other 26 identities are deterministically divided into 16 training, 5 validation, and 5 test identities. An impostor identity never appears in more than one partition. All 34 windows of an assigned impostor identity stay in that partition; the authorized user's own windows retain the chronological 20/7/7 split. Ten identity partitions measure sensitivity to which attackers are held out. They are not independent replications and do not create new recording sessions.

## Interpretation boundary

The available data can assess within-recording separation and sensitivity to feature choice, chronology, and known versus unseen impostor identities. It cannot establish persistence across electrode reapplication, days, devices, or sessions. The decisive follow-up is an enrollment/validation/test design using independently collected sessions with electrodes removed and reapplied between sessions.
