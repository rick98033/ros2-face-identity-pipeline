# Biometric data governance

This repository contains code for face detection, embeddings, matching, and
target authorization. Deployments may therefore process biometric identifiers
and images of bystanders.

Before collecting real data, define and independently review:

- informed enrollment and withdrawal of consent;
- bystander handling and camera-visible notices;
- the exact purpose for which identity may be used;
- encryption, access control, retention, deletion, and backup behavior;
- whether raw frames, crops, embeddings, scores, or decision logs are retained;
- false-accept and false-reject evaluation for the deployment population;
- behavior under ambiguity, demographic performance differences, occlusion,
  replay attempts, presentation attacks, and model failure;
- human review and recourse;
- applicable biometric, privacy, accessibility, and consumer-protection law.

No real face image, crop, embedding, or enrollment database is included in this
publication candidate. Future fixtures should be synthetic or accompanied by
documented permission and redistribution rights.
