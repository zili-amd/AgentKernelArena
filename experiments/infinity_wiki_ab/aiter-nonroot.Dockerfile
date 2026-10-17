# The Aiter image with AITER's root-only paths opened to the run's user, nothing else changed. As another user,
# AITER cannot write its JIT directory, so `import aiter` copies it into $HOME and fails on FlyDSL caches only root
# can read, and it cannot lock the merged tuned configs under /tmp/aiter_configs. Root runs need none of this.
#   docker build --build-arg BASE=<phases.mi300_aiter.docker_image> -t aka-wiki-ab/sglang-aiter-dbd8bf5-flydsl032:nonroot - < aiter-nonroot.Dockerfile
ARG BASE
FROM ${BASE}
USER root
RUN chmod -R a+rwX /tmp/aiter_configs /sgl-workspace/aiter/aiter/jit
