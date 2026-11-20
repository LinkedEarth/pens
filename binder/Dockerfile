# myBinder builds from this file. It reuses the tested image that
# .github/workflows/build-container.yml built and pushed for a release, so the
# environment is never re-resolved and cannot drift.
# At each release, update the tag below to the new release name.
FROM quay.io/julieneg/pens:2026.11.19

# Refresh the repository contents (notebooks, data) from the commit being launched.
COPY --chown=jovyan:jovyan . /home/jovyan/
