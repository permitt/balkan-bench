# GCP Security

## Secrets

**`HF_OFFICIAL_TOKEN`** is the only secret the BalkanBench GCP flow handles.
It gates read access to the private-labels HF repo (`permitt/superglue-private`)
and write access to the public-labels HF repo during publish.

Storage: GCP Secret Manager, secret name `balkanbench-hf-official-token`
(override with `HF_SECRET_NAME`). Rotation: on compromise + every 90 days.

Never:
- Bake the token into the Docker image.
- Log the token (the launcher redacts it).
- Commit the token to git (verified by a simple grep in `release-check.yml`).

## Service-account IAM

The VM's default service account needs, at minimum:
- `roles/secretmanager.secretAccessor` on the token secret.
- `roles/storage.objectCreator` on the artifacts bucket.
- `roles/artifactregistry.reader` on the Docker image repo.

The launching identity additionally needs `roles/compute.instanceAdmin.v1`.

## Audit logs

GCP Cloud Audit Logs capture every VM create + delete and every Secret
Manager access. Retain at least 90 days. This is the audit trail that
backs the benchmark-integrity claim in the methodology docs.

## Private-label isolation

- The private labels HF repo is not mirrored to GCS.
- Official scoring reads the private labels just-in-time inside the VM
  via `HF_OFFICIAL_TOKEN` and discards them with the VM.
- `balkanbench score` refuses to run without the token in env; no silent
  "eval on public data" fallback.

## Reporting vulnerabilities

Security contact: [perovicmitar@gmail.com](mailto:perovicmitar@gmail.com)
(pre-launch). Post-launch this moves to a dedicated address published in
`SECURITY.md`.
