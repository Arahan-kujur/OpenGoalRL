"""GRF gym API compatibility helpers (no gfootball import required)."""


def patch_grf_api(env):
    """Fix gym old/new API mismatch inside gfootball's wrapper chain."""
    current = env
    while True:
        orig_reset = current.reset
        orig_step = current.step

        def _make_patched_reset(orig):
            def _patched(**kwargs):
                result = orig(**kwargs)
                if isinstance(result, tuple):
                    return result
                return result, {}
            return _patched

        def _make_patched_step(orig):
            def _patched(action):
                result = orig(action)
                if len(result) == 5:
                    return result
                obs, reward, done, info = result
                return obs, reward, done, False, info
            return _patched

        current.reset = _make_patched_reset(orig_reset)
        current.step = _make_patched_step(orig_step)

        if hasattr(current, "env"):
            current = current.env
        else:
            break
