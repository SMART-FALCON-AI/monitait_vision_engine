# wheels/ — offline Python packages for MVE

## Why this exists

Some MVE sites (vteam12, kiancord, khoy) run behind a network where the
container can't reach `files.pythonhosted.org` to `pip install`. The
`vision_engine/startup.sh` entrypoint checks for missing modules on
every boot and installs from wheels found in this directory.

Currently the only auto-installed wheel is `pypylon` (Basler USB3 Vision
SDK) — required for pro cameras. Sites without Basler can leave this dir
empty; MVE boots normally with USB / IP cameras.

## Layout

    wheels/
      pypylon-26.6-cp39-abi3-manylinux_2_31_x86_64.whl   # NOT in git (86 MB)
      README.md                                            # this file

## How to update pypylon on a site

From any machine that CAN reach pypi:

    python -m pip download pypylon --platform manylinux_2_31_x86_64 \
                                    --python-version 310 \
                                    --only-binary=:all: \
                                    --no-deps \
                                    -d ./wheels/

Then `scp` the new `.whl` to the site's `wheels/` directory. On the next
`docker restart monitait_vision_engine` (or `docker compose up`),
`startup.sh` will detect the missing/updated version and re-install.

## Why not just add pypylon to `vision_engine/requirements.txt`?

Because the image build box also often can't reach pypi. Baking pypylon
into the image would work in a lab but fail at every restricted site.
The offline-wheel approach is portable — the wheel is a self-contained
artifact you carry from a build machine to each target site once, and it
survives every recreate.
