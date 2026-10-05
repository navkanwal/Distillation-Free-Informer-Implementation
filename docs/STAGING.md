# Source-only first-commit file set

No Git repository has been initialized, and no files have been staged, committed or pushed.

[first-commit-files.txt](first-commit-files.txt) is the final source/configuration/documentation file list for the initial public commit. Paths are relative to this standalone root. [source-review-files.txt](source-review-files.txt) contains the same list for review. These lists include their own paths and are review artifacts, not staging actions.

All ten paths in [artifact-decision-files.txt](artifact-decision-files.txt) are excluded from both lists and protected by exact root-anchored `.gitignore` entries. Keep local copies for review; do not force-add them. No global JSON, metadata, checkpoint or model-file exclusions were introduced. Normal local data, secret, environment, dependency, build and cache exclusions still apply.

This first release distributes source only. Install dependencies, acquire the CSV separately and follow [README Training](../README.md#training) to regenerate `energy_public_v1`. API startup fails until a complete valid public-profile bundle is available. The frontend builds without the withheld fitted catalog and obtains categories from the validated API. Internal/private artifacts are never a fallback.

The documentation preserves the verified release candidate's evaluation metrics. Historical runtime/browser/equivalence reports refer to that candidate with artifacts present; [source-only checks](verification/source-only-release-checks.json) describe the earlier preparation snapshot, before the current documentation cleanup. Their file counts and test results are historical, not a fresh validation of the current tree.

Associated-publication metadata still needs owner completion. The artifact decision for this initial commit is settled: omit the ten files. Any later artifact redistribution requires a separate decision.

`.gitattributes` preserves original trainer/model source bytes and local bundle bytes for provenance checks. `.gitignore` does not untrack already tracked files; this standalone directory has no `.git` directory and nothing is tracked here.
